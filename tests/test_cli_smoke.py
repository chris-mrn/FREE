"""CLI smoke tests — verifies main.py subcommand help strings and 2D training."""
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run(args, **kwargs):
    return subprocess.run(
        [PYTHON] + args,
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': REPO},
        cwd=REPO,
        **kwargs
    )


def test_main_help():
    r = run(['main.py', '--help'])
    assert r.returncode == 0, r.stderr
    assert 'train' in r.stdout
    assert 'evaluate' in r.stdout


def test_train_help():
    r = run(['main.py', 'train', '--help'])
    assert r.returncode == 0, r.stderr
    assert '--path' in r.stdout
    assert '--dataset' in r.stdout
    assert '--t_mode' in r.stdout


def test_evaluate_help():
    r = run(['main.py', 'evaluate', '--help'])
    assert r.returncode == 0, r.stderr
    assert '--ckpt_dir' in r.stdout


def test_compare_help():
    r = run(['main.py', 'compare', '--help'])
    assert r.returncode == 0, r.stderr
    assert '--speed_dir' in r.stdout


def test_analyze_help():
    r = run(['main.py', 'analyze', '--help'])
    assert r.returncode == 0, r.stderr
    assert '--mode' in r.stdout


def test_2d_smoke():
    r = run([
        'main.py', 'train',
        '--path', 'linear', '--dataset', '8gaussians',
        '--out_dir', '/tmp/test_cli_smoke',
        '--total_steps', '5', '--eval_every', '5',
        '--resume', 'disabled', '--seed', '0',
    ], timeout=120)
    assert r.returncode == 0, r.stderr + r.stdout
    assert 'Done' in r.stdout
