# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared caller/profile data contract and host-only failure reporting."""

import base64
import copy
import email.message
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
import urllib.error
from unittest.mock import patch

import yaml

from scripts import agent_review_contract as contract

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.load((ROOT / '.github/workflows/agent-review-reusable.yaml').read_text(), Loader=yaml.BaseLoader)
SHA = 'a' * 40


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Resolve platform temp symlinks such as macOS /var -> /private/var; the reader refuses symlinked paths.
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'profile.toml'
        self.good = '[review]\ninstructions="Review correctness"\n[ignore]\nglob=["**/vendor/**"]\n'
        self.path.write_text(self.good)

    def test_only_instructions_and_exclusions_are_supported(self):
        data, digest = contract.load_profile(self.path)
        self.assertEqual(hashlib.sha256(self.good.encode()).hexdigest(), digest)
        self.assertEqual(['**/vendor/**'], data['ignore']['glob'])
        for attack in ('\n[config]\nmodel="other"', '\n[github]\nuser_token="fixture"',
                       '\n[litellm]\nenable_callbacks=true', '\n[review.extra]\nkey=true'):
            with self.subTest(attack=attack):
                self.path.write_text(self.good + attack)
                with self.assertRaises(ValueError):
                    contract.load_profile(self.path)
        for attack in (self.good.replace('instructions=', 'unknown='),
                       self.good.replace('"Review correctness"', 'false'),
                       self.good.replace('"Review correctness"', '""'),
                       self.good.replace('["**/vendor/**"]', '"**/vendor/**"'),
                       self.good.replace('["**/vendor/**"]', '[false]'),
                       self.good.replace('["**/vendor/**"]', '[""]'),
                       '[review]\ninstructions="only one section"', 'not toml'):
            with self.subTest(attack=attack):
                self.path.write_text(attack)
                with self.assertRaises(ValueError):
                    contract.load_profile(self.path)

    def test_reader_rejects_oversize_nonregular_and_symlink_components(self):
        self.path.write_bytes(b'x' * contract.MAX_DATA_BYTES)
        self.assertEqual(contract.MAX_DATA_BYTES, len(contract.read_data(self.path)))
        self.path.write_bytes(b'x' * (contract.MAX_DATA_BYTES + 1))
        with self.assertRaises(ValueError):
            contract.read_data(self.path)
        # Valid content, so only path refusal can make the following reads fail.
        self.path.write_text(self.good)
        self.assertEqual(self.good, contract.read_data(self.path))
        with self.assertRaises(ValueError):
            contract.read_data(self.root / '..' / self.root.name / 'profile.toml')
        link = self.root / 'link'
        link.symlink_to(self.path)
        directory = self.root / 'directory'
        directory.symlink_to(self.root, target_is_directory=True)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        for path in (link, directory / 'profile.toml', self.root, fifo):
            with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                contract.read_data(path)
        self.path.write_bytes(b'\xff')
        with self.assertRaises(UnicodeError):
            contract.read_data(self.path)

    def test_own_profile_reviews_engine_source_and_workflows(self):
        import fnmatch
        data, _ = contract.load_profile(ROOT / '.github/agent-review.toml')
        globs = data['ignore']['glob']
        for path in ('scripts/review_engine.py', 'scripts/agent_review_contract.py',
                     '.github/workflows/agent-review-reusable.yaml', '.github/pr-agent.toml',
                     'examples/pipelock/agent-review.toml'):
            with self.subTest(path=path):
                self.assertFalse(any(fnmatch.fnmatch(path, g) for g in globs))

    def test_pipelock_example_profile_preserves_included_source_and_declared_exclusions(self):
        import fnmatch
        data, _ = contract.load_profile(ROOT / 'examples/pipelock/agent-review.toml')
        globs = data['ignore']['glob']
        for path, excluded in [('internal/scanner/scanner.go', False), ('scripts/review_engine.py', False),
                               ('internal/scanner/testdata/fixture.json', True), ('api/example.pb.go', True),
                               ('web/package-lock.json', True)]:
            with self.subTest(path=path):
                self.assertEqual(excluded, any(fnmatch.fnmatch(path, g) for g in globs))
        self.assertIn('Core literal-IP SSRF', data['review']['instructions'])


class CallerTests(unittest.TestCase):
    def test_exact_local_and_pinned_remote_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'caller.yaml'
            path.write_text(contract.EXPECTED_CALLER)
            contract.validate_caller(path, SHA, allow_local=True)
            with self.assertRaises(ValueError):
                contract.validate_caller(path, SHA)
            remote = contract.EXPECTED_CALLER.replace(contract.LOCAL_CALL, contract.REMOTE_CALL + SHA + ' # main')
            path.write_text(remote)
            contract.validate_caller(path, SHA)
            mutations = [remote.replace(SHA, 'main'), remote.replace(SHA, 'b' * 40),
                         remote.replace("github.event.comment.user.id == 142104046 &&", ''),
                         remote.replace("github.run_attempt == '1' &&", ''),
                         remote.replace("github.event.comment.author_association == 'MEMBER' &&", ''),
                         remote.replace("github.event.comment.user.type == 'User' &&", ''),
                         remote.replace('["/review","/review deep","/review codex","/review codex deep"]',
                                        '["/review","/review deep","/review codex","/other"]'),
                         remote.replace('types: [created]', 'types: [created, edited]'),
                         remote.replace('pull-requests: write', 'pull-requests: read'),
                         remote.replace('    secrets:\n', '    secrets: inherit\n    ignored:\n'),
                         remote + '\n    env:\n      PYTHONPATH: proposed\n']
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    path.write_text(mutation)
                    with self.assertRaises(ValueError):
                        contract.validate_caller(path, SHA)
            path.write_text(remote)
            with self.assertRaises(ValueError):
                contract.validate_caller(path, 'main')
        self.assertEqual(contract.EXPECTED_CALLER, (ROOT / '.github/workflows/agent-review.yaml').read_text())

    def test_example_caller_is_the_remote_template(self):
        example = (ROOT / 'examples/pipelock/agent-review.yaml').read_text()
        placeholder = '<full-40-char-commit-sha>'
        self.assertEqual(contract.EXPECTED_CALLER.replace(contract.LOCAL_CALL, contract.REMOTE_CALL + placeholder + ' # main'),
                         example)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'caller.yaml'
            path.write_text(example.replace(placeholder, SHA))
            contract.validate_caller(path, SHA)
            path.write_text(example)
            with self.assertRaises(ValueError):
                contract.validate_caller(path, SHA)

    def test_contract_binds_the_running_caller_file(self):
        repo = 'example/project'
        contract.validate_caller_ref(f'{repo}/.github/workflows/agent-review.yaml@refs/heads/main', repo)
        for ref in (None, '', f'{repo}/.github/workflows/agent-review.yaml@',
                    f'{repo}/.github/workflows/other.yaml@refs/heads/main',
                    f'{repo}/.github/workflows/agent-review.yaml.bak@refs/heads/main',
                    f'{repo}/.github/workflows/agent-review.yaml@decoy.yaml@refs/heads/main',
                    f'{repo}/.github/workflows/agent-review.yaml@refs/heads/x.yaml@refs/heads/main',
                    f'{repo}/.github/workflows/agent-review.yaml@refs/tags/v1',
                    f'{repo}/.github/workflows/agent-review.yaml@refs/heads/',
                    f'{repo}/.github/workflows/agent-review.yaml@refs/heads/main x',
                    f'example/other/.github/workflows/agent-review.yaml@refs/heads/main',
                    f'x{repo}/.github/workflows/agent-review.yaml@refs/heads/main'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                contract.validate_caller_ref(ref, repo)
        for bad_repo in (None, '', 'example', 'example/project/extra'):
            with self.subTest(repo=bad_repo), self.assertRaises(ValueError):
                contract.validate_caller_ref(f'{bad_repo}/.github/workflows/agent-review.yaml@refs/heads/main', bad_repo)
        step = next(s for s in WORKFLOW['jobs']['review']['steps'] if s.get('name') == 'Validate the trusted caller and restricted profile before inference')
        self.assertEqual('${{ github.workflow_ref }}', step['env']['CALLER_WORKFLOW_REF'])
        # A matching caller file and profile still fail when another file is running.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / '.github/workflows').mkdir(parents=True)
            (root / '.github/workflows/agent-review.yaml').write_text(
                contract.EXPECTED_CALLER.replace(contract.LOCAL_CALL, contract.REMOTE_CALL + SHA + ' # main'))
            (root / '.github/agent-review.toml').write_text('[review]\ninstructions="Review correctness"\n[ignore]\nglob=[]\n')
            argv = ['agent_review_contract.py', '--root', str(root), '--source-sha', SHA]
            for ref, code in ((f'{repo}/.github/workflows/agent-review.yaml@refs/heads/main', None),
                              (f'{repo}/.github/workflows/decoy-caller.yaml@refs/heads/main', 1)):
                with (self.subTest(ref=ref), patch('sys.argv', argv),
                      patch.dict(os.environ, {'GITHUB_REPOSITORY': repo, 'CALLER_WORKFLOW_REF': ref}),
                      patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO())):
                    if code is None:
                        contract.main()
                    else:
                        with self.assertRaises(SystemExit) as raised:
                            contract.main()
                        self.assertEqual(code, raised.exception.code)


    def test_every_job_repeats_the_exact_admission_gate(self):
        gate = WORKFLOW['jobs']['admit']['if']
        reply_gate = WORKFLOW['jobs']['reply']['if']
        self.assertEqual(f'({gate}) || ({reply_gate})', WORKFLOW['jobs']['source']['if'])
        self.assertTrue(WORKFLOW['jobs']['notify-failure']['if'].startswith('always() && ' + gate + ' && ('))


    def test_engine_dependencies_install_before_any_step_that_imports_them(self):
        # Steps that import the pinned GitHub client (reply-policy resolution and review_jobs.py)
        # must follow the hash-pinned install in their job; otherwise the import fails and the
        # contract's fixed "Invalid review caller or profile" message hides the cause.
        checked = 0
        for name, job in WORKFLOW['jobs'].items():
            installed = False
            for step in job.get('steps', []):
                run = step.get('run', '')
                if 'pip install --require-hashes -r trusted-agent-review/.github/requirements-runtime.txt' in run:
                    installed = True
                if '--resolve-reply-policy' in run or 'trusted-agent-review/scripts/review_jobs.py' in run:
                    checked += 1
                    self.assertTrue(installed, f"{name}: {step.get('name')} runs before the dependency install")
        self.assertGreaterEqual(checked, 4)


class SourceBindingTests(unittest.TestCase):
    """The helper must bind the engine checkout to this repository's loaded revision."""

    HELPER = yaml.load((ROOT / '.github/workflows/agent-review-source.yaml').read_text(), Loader=yaml.BaseLoader)

    def bind(self, **overrides):
        step = self.HELPER['jobs']['source']['steps'][0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            output.write_text('')
            env = {'PATH': os.environ['PATH'], 'GITHUB_OUTPUT': str(output), 'WORKFLOW_SHA': SHA,
                   'WORKFLOW_REPOSITORY': 'pipelab-org/agent-review',
                   'WORKFLOW_FILE_PATH': '.github/workflows/agent-review-source.yaml', 'REVIEWER_SHA': ''}
            env.update(overrides)
            result = subprocess.run(['bash', '-c', step['run']], env=env, capture_output=True, text=True, check=False)
            return result.returncode, output.read_text()

    def test_helper_is_credential_free(self):
        self.assertEqual({}, self.HELPER['permissions'])
        self.assertEqual({}, self.HELPER['jobs']['source']['permissions'])
        self.assertNotIn('secrets', self.HELPER['on']['workflow_call'])
        self.assertNotIn('uses', json.dumps(self.HELPER['jobs']['source']['steps']))

    def test_engine_checkout_names_the_repository_the_helper_requires(self):
        checkout = next(s for s in WORKFLOW['jobs']['review']['steps'] if s.get('name') == 'Check out immutable agent-review source')
        self.assertEqual('pipelab-org/agent-review', checkout['with']['repository'])
        self.assertIn('"${WORKFLOW_REPOSITORY}" != pipelab-org/agent-review',
                      self.HELPER['jobs']['source']['steps'][0]['run'])

    def test_loaded_identity_binds_and_every_other_identity_fails_closed(self):
        self.assertEqual((0, f'reviewer_sha={SHA}\n'), self.bind())
        self.assertEqual((0, f'reviewer_sha={SHA}\n'), self.bind(REVIEWER_SHA=SHA))
        for overrides in ({'WORKFLOW_REPOSITORY': 'pipelab-org/pipelock'},
                          {'WORKFLOW_REPOSITORY': 'example/agent-review'},
                          {'WORKFLOW_FILE_PATH': '.github/workflows/pr-review-source.yaml'},
                          {'WORKFLOW_FILE_PATH': '.github/workflows/agent-review-reusable.yaml'},
                          {'WORKFLOW_SHA': 'main'}, {'WORKFLOW_SHA': 'A' * 40}, {'WORKFLOW_SHA': ''},
                          {'REVIEWER_SHA': 'b' * 40}):
            with self.subTest(overrides=overrides):
                code, output = self.bind(**overrides)
                self.assertNotEqual(0, code)
                self.assertEqual('', output)


class ReplySourceTests(unittest.TestCase):
    """Inline self-calls cannot select executable Python from the PR merge."""

    def resolve(self, repository, responses=(), loaded=SHA):
        step = WORKFLOW['jobs']['reply']['steps'][0]
        self.assertEqual('reply-source', step['id'])
        script = step['run'].split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        calls = []

        def request(req):
            calls.append(req.full_url)
            value = responses[len(calls) - 1]
            if isinstance(value, Exception):
                raise value
            return io.BytesIO(json.dumps(value).encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            env = {'GITHUB_REPOSITORY': repository, 'GITHUB_TOKEN': 'fixture',
                   'LOADED_REVIEW_SOURCE_SHA': loaded, 'GITHUB_OUTPUT': str(output)}
            with patch.dict(os.environ, env), patch('urllib.request.urlopen', side_effect=request):
                try:
                    exec(compile(script, '<trusted-reply-source>', 'exec'), {})
                except SystemExit as error:
                    return str(error), output.read_text() if output.exists() else '', calls
            return None, output.read_text(), calls

    def test_local_inline_merge_source_is_replaced_before_executable_checkout(self):
        default = 'b' * 40
        error, output, calls = self.resolve('pipelab-org/agent-review',
                                           [{'default_branch': 'release/review'}, {'sha': default}])
        self.assertIsNone(error)
        self.assertEqual(f'reviewer_sha={default}\n', output)
        self.assertNotIn(SHA, output)
        self.assertEqual(['https://api.github.com/repos/pipelab-org/agent-review',
                          'https://api.github.com/repos/pipelab-org/agent-review/commits/release%2Freview'], calls)

    def test_remote_caller_retains_immutable_pin_without_api_access(self):
        self.assertEqual((None, f'reviewer_sha={SHA}\n', []), self.resolve('example/project'))

    def test_missing_failed_or_malformed_default_source_never_exports_a_checkout(self):
        for responses in ([RuntimeError('private failure detail')], [{}],
                          [{'default_branch': 'main'}, RuntimeError('private failure detail')],
                          [{'default_branch': 'main'}, {'sha': 'main'}],
                          [{'default_branch': 'main'}, {'sha': None}]):
            with self.subTest(responses=responses):
                error, output, _ = self.resolve('pipelab-org/agent-review', responses)
                self.assertEqual('Trusted reply source unavailable; no repository code executed', error)
                self.assertEqual('', output)
        error, output, calls = self.resolve('example/project', loaded='main')
        self.assertIsNotNone(error)
        self.assertEqual('', output)
        self.assertEqual([], calls)

    def test_every_reply_source_consumer_uses_the_precheckout_selection(self):
        steps = WORKFLOW['jobs']['reply']['steps']
        self.assertNotIn('uses', steps[0])
        self.assertEqual({'GITHUB_TOKEN', 'LOADED_REVIEW_SOURCE_SHA'}, set(steps[0]['env']))
        selected = '${{ steps.reply-source.outputs.reviewer_sha }}'
        self.assertEqual(selected, next(s for s in steps if s.get('name') == 'Check out immutable agent-review source')['with']['ref'])
        self.assertEqual('pipelab-org/agent-review', next(s for s in steps if s.get('name') == 'Check out immutable agent-review source')['with']['repository'])
        consumers = [s['env']['REVIEW_SOURCE_SHA'] for s in steps if 'REVIEW_SOURCE_SHA' in s.get('env', {})]
        self.assertTrue(consumers)
        self.assertEqual({selected}, set(consumers))
        self.assertEqual('${{ needs.source.outputs.reviewer_sha }}',
                         next(s for s in WORKFLOW['jobs']['review']['steps'] if s.get('name') == 'Check out immutable agent-review source')['with']['ref'])


class LocalCallerTests(unittest.TestCase):
    """Only the engine repository may load the reusable workflow by local path."""

    def run_main(self, repository, caller):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / '.github/workflows').mkdir(parents=True)
            (root / '.github/workflows/agent-review.yaml').write_text(caller)
            (root / '.github/agent-review.toml').write_text('[review]\ninstructions="Review correctness"\n[ignore]\nglob=[]\n')
            argv = ['agent_review_contract.py', '--root', str(root), '--source-sha', SHA]
            ref = f'{repository}/.github/workflows/agent-review.yaml@refs/heads/main'
            with (patch('sys.argv', argv),
                  patch.dict(os.environ, {'GITHUB_REPOSITORY': repository, 'CALLER_WORKFLOW_REF': ref}),
                  patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO())):
                try:
                    contract.main()
                except SystemExit as raised:
                    return raised.code
                return 0

    def test_local_form_only_in_engine_repository(self):
        self.assertEqual(0, self.run_main('pipelab-org/agent-review', contract.EXPECTED_CALLER))
        for repository in ('pipelab-org/pipelock', 'example/project', 'pipelab-org/agent-review-fork'):
            with self.subTest(repository=repository):
                self.assertEqual(1, self.run_main(repository, contract.EXPECTED_CALLER))

    def test_remote_form_names_only_the_engine_repository(self):
        remote = contract.EXPECTED_CALLER.replace(contract.LOCAL_CALL, contract.REMOTE_CALL + SHA + ' # main')
        self.assertIn('uses: pipelab-org/agent-review/.github/workflows/agent-review-reusable.yaml@' + SHA + ' # main', remote)
        self.assertEqual(0, self.run_main('pipelab-org/pipelock', remote))
        legacy = remote.replace('pipelab-org/agent-review/', 'pipelab-org/pipelock/')
        self.assertEqual(1, self.run_main('pipelab-org/pipelock', legacy))


class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.event_path = Path(self.temp.name).resolve() / 'event.json'
        self.event = {'action': 'created', 'repository': {'full_name': 'example/project'},
                      'issue': {'number': 1, 'pull_request': {'url': 'untrusted'}},
                      'comment': {'body': '/review', 'author_association': 'MEMBER',
                                  'user': {'login': 'luckyPipewrench', 'id': 142104046, 'type': 'User'}}}
        self.env = {'GITHUB_EVENT_PATH': str(self.event_path), 'GITHUB_EVENT_NAME': 'issue_comment',
                    'GITHUB_REPOSITORY': 'example/project', 'GITHUB_ACTOR': 'luckyPipewrench',
                    'GITHUB_ACTOR_ID': '142104046', 'GITHUB_TRIGGERING_ACTOR': 'luckyPipewrench',
                    'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_RUN_ID': '123', 'GH_TOKEN': 'fixture',
                    'REVIEW_HEAD': 'a' * 40, 'REVIEW_BASE': 'b' * 40, 'ADMISSION_RESULT': 'success',
                    'REVIEW_OUTCOME_PATH': str(Path(self.temp.name).resolve() / 'notifier' / 'outcome.json'),
                    'GITHUB_STEP_SUMMARY': str(Path(self.temp.name).resolve() / 'summary.md')}
        self.pr = {'state': 'open', 'head': {'repo': {'full_name': 'example/project', 'fork': False}, 'sha': 'a' * 40},
                   'base': {'repo': {'full_name': 'example/project'}, 'sha': 'b' * 40}}
        self.comments = []
        self.calls = []
        self.fail_method = None

    def execute(self, error=None):
        script = WORKFLOW['jobs']['notify-failure']['steps'][0]['run'].split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        self.event_path.write_text(json.dumps(self.event))

        def request(req):
            self.calls.append((req.method, req.full_url, json.loads(req.data) if req.data else None))
            if error and self.fail_method in (None, req.method):
                raise error
            if req.full_url.endswith('/pulls/1'):
                result = self.pr
            elif req.method == 'GET':
                result = self.comments
            else:
                result = {'id': 7}
            return io.BytesIO(json.dumps(result).encode())

        with patch.dict(os.environ, self.env), patch('urllib.request.urlopen', side_effect=request):
            exec(compile(script, '<trusted-reporter>', 'exec'), {})

    def test_reporter_is_host_only_and_minimally_privileged(self):
        job = WORKFLOW['jobs']['notify-failure']
        self.assertEqual(['source', 'admit', 'review', 'publish'], job['needs'])
        self.assertIn('always()', job['if'])
        self.assertEqual({'pull-requests': 'write'}, job['permissions'])
        self.assertNotIn('container', job)
        text = json.dumps(job)
        for denied in ('secrets.', 'checkout', 'download-artifact', 'review_jobs.py'):
            self.assertNotIn(denied, text)

    def test_notification_outcome_is_retained_after_a_failed_step(self):
        job = WORKFLOW['jobs']['notify-failure']
        upload = job['steps'][-1]
        self.assertEqual('always()', upload['if'])
        self.assertTrue(upload['uses'].startswith('actions/upload-artifact@'))
        self.assertEqual('agent-review-notifier-outcome', upload['with']['name'])
        # Diagnostics survive a re-run and cannot turn a run red themselves.
        self.assertEqual(('true', 'true'), (upload['continue-on-error'], upload['with']['overwrite']))
        self.assertEqual(job['steps'][0]['env']['REVIEW_OUTCOME_PATH'], upload['with']['path'])

    def test_failure_is_visible_and_existing_per_run_bot_notice_is_updated(self):
        self.execute()
        self.assertEqual('POST', self.calls[-1][0])
        body = self.calls[-1][2]['body']
        self.assertIn('incomplete', body)
        self.assertIn('/actions/runs/123', body)
        self.assertNotIn('fixture', body)
        self.assertIn('a' * 40, body)
        self.comments = [{'id': 7, 'body': body, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}}]
        self.execute()
        self.assertEqual('PATCH', self.calls[-1][0])
        self.comments[0]['user']['login'] = 'contributor'
        self.execute()
        self.assertEqual('POST', self.calls[-1][0])

    def test_admission_and_stale_snapshot_have_fixed_messages(self):
        self.env.update(REVIEW_HEAD='', REVIEW_BASE='', ADMISSION_RESULT='failure')
        self.execute()
        self.assertIn('never started', self.calls[-1][2]['body'])
        self.env.update(REVIEW_HEAD='c' * 40, REVIEW_BASE='b' * 40, ADMISSION_RESULT='success')
        self.execute()
        self.assertIn('commits changed', self.calls[-1][2]['body'])
        with self.assertRaisesRegex(SystemExit, '^Review failure notification failed; inspect Actions status$'):
            self.execute(error=RuntimeError('private provider detail'))

    def http_error(self, status, **headers):
        message = email.message.Message()
        for name, value in headers.items():
            message[name.replace('_', '-')] = value
        return urllib.error.HTTPError('https://api.github.com/repos/example/project/SECRET-URL', status,
                                      'SECRET-REASON', message, io.BytesIO(b'{"message":"SECRET-BODY"}'))

    def outcome(self):
        return json.loads(Path(self.env['REVIEW_OUTCOME_PATH']).read_text())

    def test_failed_notification_keeps_a_machine_readable_outcome_when_no_comment_is_possible(self):
        # The same token that was rate limited cannot comment, so the record is local.
        reset = int(time.time()) + 1800
        error = self.http_error(403, x_ratelimit_remaining='0', x_ratelimit_limit='1000',
                                x_ratelimit_reset=str(reset), x_ratelimit_resource='core',
                                Authorization='Bearer SECRET-TOKEN', x_github_request_id='SECRET-ID')
        stderr = io.StringIO()
        with patch('sys.stderr', stderr), self.assertRaisesRegex(
                SystemExit, '^Review failure notification failed; inspect Actions status$'):
            self.execute(error=error)
        outcome = self.outcome()
        self.assertEqual('agent-review-outcome:v1', outcome['schema'])
        self.assertEqual('failed', outcome['result'])
        failure = outcome['failure']
        self.assertEqual(('notification', 'read_pull_request', 'rate_limit_primary', 403, 'HTTPError'),
                         (failure['phase'], failure['stage'], failure['category'], failure['status'], failure['exception']))
        self.assertEqual({'limit': 1000, 'remaining': 0, 'reset': reset, 'resource': 'core'},
                         {k: v for k, v in failure['rate_limit'].items() if k != 'reset_in_seconds'})
        self.assertTrue(0 < failure['rate_limit']['reset_in_seconds'] <= 1800)
        summary = Path(self.env['GITHUB_STEP_SUMMARY']).read_text()
        self.assertIn('rate_limit_primary', summary)
        self.assertIn('read_pull_request', summary)
        self.assertEqual(0o600, Path(self.env['REVIEW_OUTCOME_PATH']).stat().st_mode & 0o777)
        for text in (json.dumps(outcome), summary, stderr.getvalue()):
            for secret in ('SECRET', 'fixture', 'api.github.com', 'example/project'):
                self.assertNotIn(secret, text)

    def test_failure_stage_and_category_follow_the_failing_operation(self):
        self.fail_method = 'POST'
        with self.assertRaises(SystemExit):
            self.execute(error=self.http_error(502))
        self.assertEqual(('create_comment', 'server_error', 502),
                         tuple(self.outcome()['failure'][k] for k in ('stage', 'category', 'status')))
        self.fail_method = 'PATCH'
        self.comments = [{'id': 7, 'body': '<!-- agent-review-notice:v1 run=123 -->\nold',
                          'user': {'login': 'github-actions[bot]', 'type': 'Bot'}}]
        with self.assertRaises(SystemExit):
            self.execute(error=self.http_error(403))
        self.assertEqual(('update_comment', 'forbidden'),
                         tuple(self.outcome()['failure'][k] for k in ('stage', 'category')))
        self.fail_method = None
        with self.assertRaises(SystemExit):
            self.execute(error=urllib.error.URLError('SECRET-NETWORK'))
        failure = self.outcome()['failure']
        self.assertEqual('network', failure['category'])
        self.assertNotIn('status', failure)
        with self.assertRaises(SystemExit):
            self.execute(error=RuntimeError('private provider detail'))
        self.assertEqual('unexpected', self.outcome()['failure']['category'])
        self.assertNotIn('private', Path(self.env['REVIEW_OUTCOME_PATH']).read_text())

    def test_notifier_classification_matches_the_engines_for_header_only_evidence(self):
        from scripts import review_reliability as reliability
        cases = [(403, {'x-ratelimit-remaining': '0', 'x-ratelimit-reset': '4102444800'}),
                 (403, {'retry-after': '60'}), (403, {'retry-after': 'Wed, 21 Oct 2026 07:28:00 GMT'}),
                 (429, {}), (429, {'x-ratelimit-remaining': '0'}), (403, {}), (401, {}), (404, {}),
                 (422, {}), (500, {}), (503, {'retry-after': '5'}), (418, {}),
                 (403, {'x-ratelimit-remaining': '-1', 'x-ratelimit-reset': '0'}),
                 (403, {'x-ratelimit-resource': 'Core Data', 'x-ratelimit-limit': '1e3'})]
        for status, headers in cases:
            with self.subTest(status=status, headers=headers):
                self.fail_method = None
                with self.assertRaises(SystemExit):
                    self.execute(error=self.http_error(status, **{k.replace('-', '_'): v for k, v in headers.items()}))
                failure = self.outcome()['failure']
                self.assertEqual(reliability.classify(status, headers), failure['category'])
                expected = {k: v for k, v in reliability.rate_headers(headers).items() if k != 'reset_in_seconds'}
                actual = {k: v for k, v in failure.get('rate_limit', {}).items() if k != 'reset_in_seconds'}
                self.assertEqual(expected, actual)

    def test_outcome_sinks_are_optional_and_never_change_the_failure(self):
        for name in ('REVIEW_OUTCOME_PATH', 'GITHUB_STEP_SUMMARY'):
            del self.env[name]
        with self.assertRaisesRegex(SystemExit, '^Review failure notification failed; inspect Actions status$'):
            self.execute(error=self.http_error(403, retry_after='60'))
        self.env['REVIEW_OUTCOME_PATH'] = str(Path(self.temp.name).resolve() / 'summary-dir')
        os.mkdir(self.env['REVIEW_OUTCOME_PATH'])
        with self.assertRaisesRegex(SystemExit, '^Review failure notification failed; inspect Actions status$'):
            self.execute(error=self.http_error(403, retry_after='60'))

    def test_unauthorized_edited_rerun_issue_fork_and_closed_requests_do_not_comment(self):
        original = copy.deepcopy(self.event)
        for change in ('identity', 'command', 'edited', 'issue', 'bot', 'association'):
            self.event = copy.deepcopy(original)
            if change == 'identity':
                self.event['comment']['user']['id'] = 123
            elif change == 'command':
                self.event['comment']['body'] = '/review deep extra'
            elif change == 'edited':
                self.event['action'] = 'edited'
            elif change == 'issue':
                self.event['issue'].pop('pull_request')
            elif change == 'bot':
                self.event['comment']['user']['type'] = 'Bot'
            else:
                self.event['comment']['author_association'] = 'CONTRIBUTOR'
            self.calls.clear()
            self.execute()
            self.assertEqual([], self.calls, change)
        self.event = original
        self.env['GITHUB_RUN_ATTEMPT'] = '2'
        self.calls.clear()
        self.execute()
        self.assertEqual([], self.calls)
        self.env['GITHUB_RUN_ATTEMPT'] = '1'
        for change in ('fork', 'closed', 'missing', 'other'):
            self.calls.clear()
            self.pr['state'] = 'closed' if change == 'closed' else 'open'
            self.pr['head']['repo'] = None if change == 'missing' else {
                'full_name': 'other/project' if change == 'other' else 'example/project', 'fork': change == 'fork'}
            self.execute()
            self.assertEqual(['GET'], [c[0] for c in self.calls], change)


if __name__ == '__main__':
    unittest.main()
