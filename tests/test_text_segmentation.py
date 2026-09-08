from detection.text_segmentation import SentenceAwareSegmenter


def test_long_english_text_uses_fewer_windows_than_chinese_budget() -> None:
    sentence = "This is a generated English sentence used only for window sizing. "
    text = sentence * 40
    zh_windows = SentenceAwareSegmenter(target_chars=480, max_chars=640).split(text)
    en_windows = SentenceAwareSegmenter(target_chars=1400, max_chars=1800).split(text)
    assert len(text) > 1400
    assert len(en_windows) < len(zh_windows)
    assert all(len(window.text) <= 1800 for window in en_windows)


def test_short_text_stays_in_one_window() -> None:
    windows = SentenceAwareSegmenter(target_chars=480, max_chars=640).split("short enough")
    assert len(windows) == 1
    assert windows[0].text == "short enough"
