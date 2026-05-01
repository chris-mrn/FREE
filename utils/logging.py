"""Data loop and CSV logging utilities."""
import csv
import os


def infinite_image_loop(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, _ in loader:
            yield x
        epoch += 1


def infinite_2d_loop(dataset_name, batch_size, device):
    from data.datasets import sample_2d
    while True:
        yield sample_2d(dataset_name, batch_size).to(device)


def open_csv_logs(out_dir, start_step):
    """
    Open (or append to) loss.csv and metrics.csv in out_dir.
    Returns (loss_file, metrics_file, loss_writer, metrics_writer).
    Caller is responsible for closing both files.
    """
    loss_path    = os.path.join(out_dir, 'loss.csv')
    metrics_path = os.path.join(out_dir, 'metrics.csv')

    write_header = start_step == 0

    loss_file    = open(loss_path,    'a', newline='')
    metrics_file = open(metrics_path, 'a', newline='')

    loss_writer    = csv.writer(loss_file)
    metrics_writer = csv.writer(metrics_file)

    if write_header:
        loss_writer.writerow(['step', 'loss_raw', 'loss_ema', 't_phase'])
        metrics_writer.writerow(['step', 'fid', 'kid_mean', 'kid_std', 'is_mean', 'is_std'])

    return loss_file, metrics_file, loss_writer, metrics_writer
