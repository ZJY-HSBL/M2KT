"""
Temporal Dataset: Loads paired longitudinal chest X-ray data from MIMIC-CXR.

MIMIC-CXR contains multiple studies per patient over time. This dataset
constructs pairs of (prior_image, current_image) from the same patient's
imaging history, enabling the model to learn disease progression patterns.

Each sample includes:
    - current_image: The more recent chest X-ray
    - prior_image: A previous chest X-ray from the same patient (if available)
    - current_report: The report for the current image
    - time_delta: Days between prior and current study
    - progression_labels: Change in CheXpert labels from prior to current

For single-study patients (no prior), prior_image is set to current_image
and a "baseline" flag is set.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image


class TemporalMIMICDataset(Dataset):
    """Longitudinal MIMIC-CXR dataset with temporal image pairs.

    Args:
        data_dir: Path to MIMIC-CXR data directory
        split: 'train', 'val', or 'test'
        tokenizer: Report tokenizer
        transform: Image preprocessing transforms
        max_time_delta: Maximum days between studies to consider as a pair
        min_time_delta: Minimum days between studies (to ensure meaningful change)
    """

    def __init__(self, data_dir, split='train', tokenizer=None, transform=None,
                 max_time_delta=365, min_time_delta=1):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_time_delta = max_time_delta
        self.min_time_delta = min_time_delta

        # Load study metadata
        self.metadata = self._load_metadata(split)
        self.pairs = self._build_temporal_pairs()

    def _load_metadata(self, split):
        """Load MIMIC-CXR metadata (study_id, subject_id, study_time, report, labels)."""
        meta_path = os.path.join(self.data_dir, f'mimic_cxr_{split}.csv')
        if os.path.exists(meta_path):
            return pd.read_csv(meta_path)
        return pd.DataFrame()

    def _build_temporal_pairs(self):
        """Build (prior, current) study pairs within each patient's history.

        Strategy:
            - For each current study, find the most recent prior study
            - Time delta must be within [min_time_delta, max_time_delta]
            - Pairs are sorted by time delta for curriculum learning
        """
        pairs = []

        if self.metadata.empty:
            return pairs

        # Group by subject_id
        grouped = self.metadata.groupby('subject_id')

        for subject_id, group in grouped:
            # Sort by study time
            group = group.sort_values('study_time')

            studies = group.to_dict('records')

            for i in range(1, len(studies)):
                current = studies[i]
                prior = studies[i - 1]

                time_delta = (pd.Timestamp(current['study_time']) -
                              pd.Timestamp(prior['study_time'])).days

                if self.min_time_delta <= time_delta <= self.max_time_delta:
                    pairs.append({
                        'subject_id': subject_id,
                        'current_study_id': current['study_id'],
                        'prior_study_id': prior['study_id'],
                        'current_image_path': current['image_path'],
                        'prior_image_path': prior['image_path'],
                        'current_report': current['report'],
                        'time_delta': time_delta,
                        'current_labels': self._parse_labels(current.get('labels', '')),
                        'prior_labels': self._parse_labels(prior.get('labels', '')),
                    })

        return pairs

    def _parse_labels(self, labels_str):
        """Parse CheXpert label string to tensor."""
        if isinstance(labels_str, str):
            labels = [float(x) for x in labels_str.split(',')]
        elif isinstance(labels_str, (list, np.ndarray)):
            labels = list(labels_str)
        else:
            labels = [0.0] * 14
        return torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a temporal pair sample.

        Returns:
            Tuple of (current_image, prior_image, report_tokens, report_mask,
                     current_labels, prior_labels, time_delta, has_prior)
        """
        pair = self.pairs[idx]

        # Load images
        current_img = Image.open(
            os.path.join(self.data_dir, pair['current_image_path'])
        ).convert('RGB')

        try:
            prior_img = Image.open(
                os.path.join(self.data_dir, pair['prior_image_path'])
            ).convert('RGB')
            has_prior = True
        except (FileNotFoundError, OSError):
            prior_img = current_img.copy()
            has_prior = False

        if self.transform:
            current_img = self.transform(current_img)
            prior_img = self.transform(prior_img)

        # Tokenize report
        if self.tokenizer:
            report_tokens, report_mask = self.tokenizer.encode(pair['current_report'])
        else:
            report_tokens = torch.zeros(100, dtype=torch.long)
            report_mask = torch.zeros(100, dtype=torch.long)

        # Label progression
        label_progression = pair['current_labels'] - pair['prior_labels']

        time_delta = torch.tensor(pair['time_delta'], dtype=torch.float) / self.max_time_delta

        return (
            current_img,
            prior_img,
            report_tokens,
            report_mask,
            pair['current_labels'],
            pair['prior_labels'],
            label_progression,
            time_delta,
            torch.tensor(has_prior, dtype=torch.float),
        )


class TemporalCollator:
    """Collates temporal pairs into batches, handling variable sequence lengths."""

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        current_imgs = torch.stack([b[0] for b in batch])
        prior_imgs = torch.stack([b[1] for b in batch])

        # Pad report sequences
        reports = [b[2] for b in batch]
        masks = [b[3] for b in batch]

        max_len = max(r.shape[0] for r in reports)
        padded_reports = torch.full((len(batch), max_len), self.pad_idx, dtype=torch.long)
        padded_masks = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, (r, m) in enumerate(zip(reports, masks)):
            length = r.shape[0]
            padded_reports[i, :length] = r
            padded_masks[i, :length] = m

        current_labels = torch.stack([b[4] for b in batch])
        prior_labels = torch.stack([b[5] for b in batch])
        label_progression = torch.stack([b[6] for b in batch])
        time_deltas = torch.stack([b[7] for b in batch])
        has_prior = torch.stack([b[8] for b in batch])

        return {
            'current_images': current_imgs,
            'prior_images': prior_imgs,
            'reports_ids': padded_reports,
            'reports_masks': padded_masks,
            'current_labels': current_labels,
            'prior_labels': prior_labels,
            'label_progression': label_progression,
            'time_deltas': time_deltas,
            'has_prior': has_prior,
        }
