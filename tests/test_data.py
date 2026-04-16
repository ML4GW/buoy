from unittest.mock import patch

import pytest
import torch
from conftest import FDURATION, NUM_CHANNELS, SAMPLE_RATE

from buoy.utils.data import get_events_for_runs, slice_amplfi_data

KERNEL_LENGTH = 4.0
PSD_LENGTH = 32.0

EVENT_POSITION = KERNEL_LENGTH / 2  # event at centre of kernel


def make_data_and_times(event_position=EVENT_POSITION):
    """
    Build a strain tensor and t0/tc values sized to fit exactly one
    valid AMPLFI window with the given event_position.
    """
    total_seconds = PSD_LENGTH + KERNEL_LENGTH + FDURATION
    total_samples = int(total_seconds * SAMPLE_RATE)
    t0 = 0.0
    # choose tc so window_start lands exactly at the psd boundary
    tc = t0 + PSD_LENGTH + event_position + FDURATION / 2
    data = torch.arange(total_samples, dtype=torch.float32).expand(
        1, NUM_CHANNELS, -1
    )
    return data, t0, tc


class TestSliceAmplfiData:
    def test_output_shapes(self):
        data, t0, tc = make_data_and_times()
        psd_data, window = slice_amplfi_data(
            data=data,
            sample_rate=SAMPLE_RATE,
            t0=t0,
            tc=tc,
            kernel_length=KERNEL_LENGTH,
            event_position=EVENT_POSITION,
            psd_length=PSD_LENGTH,
            fduration=FDURATION,
        )
        assert psd_data.shape == (NUM_CHANNELS, int(PSD_LENGTH * SAMPLE_RATE))
        assert window.shape == (
            NUM_CHANNELS,
            int((KERNEL_LENGTH + FDURATION) * SAMPLE_RATE),
        )

    def test_psd_and_window_are_contiguous(self):
        """psd_data should end exactly where window begins."""
        data, t0, tc = make_data_and_times()
        psd_data, window = slice_amplfi_data(
            data=data,
            sample_rate=SAMPLE_RATE,
            t0=t0,
            tc=tc,
            kernel_length=KERNEL_LENGTH,
            event_position=EVENT_POSITION,
            psd_length=PSD_LENGTH,
            fduration=FDURATION,
        )
        # data is arange, so psd_data[-1] + 1 == window[0]
        assert torch.allclose(psd_data[:, -1] + 1, window[:, 0])

    def test_window_out_of_bounds(self):
        total_samples = int(10 * SAMPLE_RATE)
        data = torch.zeros(1, NUM_CHANNELS, total_samples)
        with pytest.raises(ValueError):
            slice_amplfi_data(
                data=data,
                sample_rate=SAMPLE_RATE,
                t0=0.0,
                tc=0.1,
                kernel_length=KERNEL_LENGTH,
                event_position=EVENT_POSITION,
                psd_length=PSD_LENGTH,
                fduration=FDURATION,
            )


class TestGetEventsForRuns:
    def test_raises_when_both_args_none(self):
        with pytest.raises(ValueError, match="At least one"):
            get_events_for_runs(None, None)

    @patch("buoy.utils.data.gwosc.datasets")
    def test_single_run(self, mock_datasets):
        mock_datasets.run_segment.return_value = (1126051217, 1137254417)
        mock_datasets.find_datasets.return_value = ["GW150914-v3"]
        result = get_events_for_runs(["O1"])
        mock_datasets.run_segment.assert_called_once_with("O1")
        mock_datasets.find_datasets.assert_called_once_with(
            type="event", segment=(1126051217, 1137254417)
        )
        assert result == ["GW150914"]

    @patch("buoy.utils.data.gwosc.datasets")
    def test_multiple_runs_deduplicates(self, mock_datasets):
        mock_datasets.run_segment.side_effect = [
            (1126051217, 1137254417),
            (1164556817, 1187733618),
        ]
        mock_datasets.find_datasets.side_effect = [
            ["GW150914-v3"],
            ["GW150914-v3", "GW170817-v2"],
        ]
        result = get_events_for_runs(["O1", "O2"])
        assert result == ["GW150914", "GW170817"]

    @patch("buoy.utils.data.gwosc.datasets")
    def test_filters_non_gw_names(self, mock_datasets):
        mock_datasets.find_datasets.return_value = [
            "190924_232654-v1",
            "GW190408_181802-v1",
        ]
        result = get_events_for_runs(observing_runs=["O3a"])
        assert result == ["GW190408_181802"]

    @patch("buoy.utils.data.gwosc.datasets")
    def test_empty_result_raises(self, mock_datasets):
        mock_datasets.find_datasets.return_value = ["190924_232654-v1"]
        with pytest.raises(ValueError, match="No public GW events found"):
            get_events_for_runs(observing_runs=["O3a"])

    @patch("buoy.utils.data.gwosc.datasets")
    def test_invalid_run_propagates_error(self, mock_datasets):
        mock_datasets.run_segment.side_effect = ValueError(
            "Run 'O9' not found."
        )
        with pytest.raises(ValueError, match="O9"):
            get_events_for_runs(["O9"])

    @patch("buoy.utils.data.gwosc.datasets")
    def test_strips_version_suffix_and_deduplicates(self, mock_datasets):
        mock_datasets.find_datasets.return_value = [
            "GW190403_051519-v1",
            "GW190403_051519-v2",
        ]
        result = get_events_for_runs(observing_runs=["O3a"])
        assert result == ["GW190403_051519"]

    @patch("buoy.utils.data.gwosc.datasets")
    def test_gps_segment(self, mock_datasets):
        mock_datasets.find_datasets.return_value = ["GW150914-v3"]
        result = get_events_for_runs(gps_segment=(1126051217, 1137254417))
        mock_datasets.find_datasets.assert_called_once_with(
            type="event", segment=(1126051217, 1137254417)
        )
        assert result == ["GW150914"]

    def test_gps_and_runs_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            get_events_for_runs(
                observing_runs=["O1"],
                gps_segment=(1126051217, 1137254417),
            )
