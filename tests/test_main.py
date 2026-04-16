import logging
import warnings
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from buoy.main import _resolve_device, main


def _make_mock_model(sample_rate=2048.0, psd_length=32.0):
    model = MagicMock()
    model.sample_rate = sample_rate
    model.psd_length = psd_length
    return model


class TestResolveDevice:
    def test_cpu_returns_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cpu_emits_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _resolve_device("cpu")
        assert any("cpu" in str(w.message).lower() for w in caught)

    def test_cuda_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="no GPU is available"):
            _resolve_device("cuda")

    def test_none_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _resolve_device(None) == "cpu"

    def test_none_uses_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _resolve_device(None) == "cuda"


class TestResolveEvents:
    """Tests for event-source validation and routing in main()."""

    def test_raises_when_no_event_source(self, tmp_path):
        with pytest.raises(ValueError, match="One of"):
            main(outdir=tmp_path, device="cpu")

    def test_raises_when_only_gps_start_provided(self, tmp_path):
        with pytest.raises(ValueError, match="together"):
            main(outdir=tmp_path, gps_start=1126051217.0, device="cpu")

    def test_raises_when_gps_and_runs_both_provided(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            main(
                outdir=tmp_path,
                observing_runs=["O1"],
                gps_start=1126051217.0,
                gps_end=1137254417.0,
                device="cpu",
            )

    @patch("buoy.main.get_events_for_runs")
    @patch("buoy.main.Amplfi")
    @patch("buoy.main.Aframe")
    def test_fetches_events_for_runs(
        self, MockAframe, MockAmplfi, mock_get_events, tmp_path
    ):
        model = _make_mock_model()
        MockAframe.return_value = model
        MockAmplfi.return_value = model
        mock_get_events.return_value = ["GW150914"]
        datadir = tmp_path / "GW150914" / "data"
        datadir.mkdir(parents=True)
        (datadir / "aframe_outputs.hdf5").touch()
        (datadir / "posterior_samples.dat").touch()

        main(outdir=tmp_path, observing_runs=["O1"], device="cpu")

        mock_get_events.assert_called_once_with(["O1"], None)

    @patch("buoy.main.get_events_for_runs")
    @patch("buoy.main.Amplfi")
    @patch("buoy.main.Aframe")
    def test_fetches_events_for_gps_range(
        self, MockAframe, MockAmplfi, mock_get_events, tmp_path
    ):
        model = _make_mock_model()
        MockAframe.return_value = model
        MockAmplfi.return_value = model
        mock_get_events.return_value = ["GW150914"]
        datadir = tmp_path / "GW150914" / "data"
        datadir.mkdir(parents=True)
        (datadir / "aframe_outputs.hdf5").touch()
        (datadir / "posterior_samples.dat").touch()

        main(
            outdir=tmp_path,
            gps_start=1126051217.0,
            gps_end=1137254417.0,
            device="cpu",
        )

        mock_get_events.assert_called_once_with(
            None, (1126051217.0, 1137254417.0)
        )

    @patch("buoy.main.Amplfi")
    @patch("buoy.main.Aframe")
    def test_warns_when_events_and_runs_both_provided(
        self, MockAframe, MockAmplfi, tmp_path, caplog
    ):
        model = _make_mock_model()
        MockAframe.return_value = model
        MockAmplfi.return_value = model
        datadir = tmp_path / "GW150914" / "data"
        datadir.mkdir(parents=True)
        (datadir / "aframe_outputs.hdf5").touch()
        (datadir / "posterior_samples.dat").touch()

        with caplog.at_level(logging.WARNING, logger="root"):
            main(
                outdir=tmp_path,
                events="GW150914",
                observing_runs=["O1"],
                device="cpu",
            )

        assert "events" in caplog.text.lower() and "run" in caplog.text.lower()


class TestMainLoop:
    """Tests for the per-event skip/force logic inside main()."""

    @pytest.fixture()
    def mock_models(self):
        model = _make_mock_model()
        with (
            patch("buoy.main.Aframe", return_value=model),
            patch("buoy.main.Amplfi", return_value=model),
        ):
            yield model

    @pytest.fixture()
    def skipped_event_dir(self, tmp_path):
        """Pre-populate output files so the event loop skips processing."""
        datadir = tmp_path / "GW150914" / "data"
        datadir.mkdir(parents=True)
        (datadir / "aframe_outputs.hdf5").touch()
        (datadir / "posterior_samples.dat").touch()
        return tmp_path

    @patch("buoy.main.Amplfi")
    @patch("buoy.main.Aframe")
    def test_raises_on_sample_rate_mismatch(
        self, MockAframe, MockAmplfi, tmp_path
    ):
        MockAframe.return_value = _make_mock_model(sample_rate=2048.0)
        MockAmplfi.side_effect = [
            _make_mock_model(sample_rate=4096.0),
            _make_mock_model(sample_rate=2048.0),
        ]
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            main(events="GW150914", outdir=tmp_path, device="cpu")

    @pytest.mark.usefixtures("mock_models")
    @patch("buoy.main.get_data")
    def test_skips_event_when_outputs_exist(
        self, mock_get_data, skipped_event_dir
    ):
        """With force=False and existing outputs, get_data is not called."""
        main(
            events="GW150914",
            outdir=skipped_event_dir,
            device="cpu",
            force=False,
        )
        mock_get_data.assert_not_called()

    @pytest.mark.usefixtures("mock_models")
    @patch("buoy.main.get_data")
    def test_skips_event_when_run_amplfi_false(self, mock_get_data, tmp_path):
        """With run_amplfi=False, only aframe output needs to exist to skip."""
        datadir = tmp_path / "GW150914" / "data"
        datadir.mkdir(parents=True)
        (datadir / "aframe_outputs.hdf5").touch()
        # posterior_samples.dat intentionally absent

        main(
            events="GW150914",
            outdir=tmp_path,
            device="cpu",
            force=False,
            run_amplfi=False,
        )
        mock_get_data.assert_not_called()

    @pytest.mark.usefixtures("mock_models")
    @patch("buoy.main.get_data")
    def test_reprocesses_event_when_force_true(
        self, mock_get_data, skipped_event_dir
    ):
        """With force=True, get_data is called even when outputs exist."""
        mock_get_data.return_value = (
            np.zeros((1, 2, 4096)),
            ["H1", "L1"],
            0.0,
            1.0,
        )
        datadir = skipped_event_dir / "GW150914" / "data"
        with h5py.File(datadir / "aframe_outputs.hdf5", "w") as f:
            f.create_dataset("times", data=np.array([0.0, 1.0]))
            f.create_dataset("ys", data=np.zeros(2))
            f.create_dataset("timing_integrated", data=np.zeros(2))
            f.create_dataset("signif_integrated", data=np.zeros(2))
            f.attrs["predicted_tc"] = 0

        main(
            events="GW150914",
            outdir=skipped_event_dir,
            device="cpu",
            force=True,
            run_aframe=False,
            run_amplfi=False,
            generate_plots=False,
        )
        mock_get_data.assert_called_once()

    @pytest.mark.usefixtures("mock_models")
    def test_parallel_execution_uses_thread_pool(self, tmp_path):
        """max_workers > 1 submits all events to a ThreadPoolExecutor."""
        for event in ["GW150914", "GW170817"]:
            datadir = tmp_path / event / "data"
            datadir.mkdir(parents=True)
            (datadir / "aframe_outputs.hdf5").touch()
            (datadir / "posterior_samples.dat").touch()

        with patch("concurrent.futures.ThreadPoolExecutor") as MockPool:
            mock_executor = MockPool.return_value.__enter__.return_value
            mock_future = MagicMock()
            mock_future.exception.return_value = None
            mock_executor.submit.return_value = mock_future

            with patch(
                "concurrent.futures.as_completed",
                return_value=[mock_future],
            ):
                main(
                    events=["GW150914", "GW170817"],
                    outdir=tmp_path,
                    device="cpu",
                    max_workers=2,
                )

        MockPool.assert_called_once_with(max_workers=2)
        assert mock_executor.submit.call_count == 2

    @pytest.mark.usefixtures("mock_models")
    @patch("buoy.main.get_data")
    def test_parallel_failure_does_not_abort_others(
        self, mock_get_data, tmp_path, caplog
    ):
        """A failing event is logged; other events still complete."""
        good = "GW150914"
        bad = "GW170817"

        good_datadir = tmp_path / good / "data"
        good_datadir.mkdir(parents=True)
        (good_datadir / "aframe_outputs.hdf5").touch()
        (good_datadir / "posterior_samples.dat").touch()

        (tmp_path / bad / "data").mkdir(parents=True)
        mock_get_data.side_effect = RuntimeError("fetch failed")

        with caplog.at_level(logging.ERROR):
            main(
                events=[good, bad],
                outdir=tmp_path,
                device="cpu",
                max_workers=2,
                force=True,
            )

        assert "GW170817" in caplog.text
        assert "fetch failed" in caplog.text
