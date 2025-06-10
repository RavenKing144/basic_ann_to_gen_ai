import re
import ast
from pathlib import Path

# Dynamically load get_youtube_video_id from the source file without importing
# the entire module (which has heavy dependencies).
def load_get_youtube_video_id():
    path = Path(__file__).resolve().parents[1] / 'langchain-projects' / 'youtube_content_summarization' / 'app.py'
    source = path.read_text()
    module = ast.parse(source)
    func_code = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'get_youtube_video_id':
            func_code = ast.get_source_segment(source, node)
            break
    namespace = {'re': re}
    exec(func_code, namespace)
    return namespace['get_youtube_video_id']

get_youtube_video_id = load_get_youtube_video_id()

import pytest

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=ABCDEFGHIJK", "ABCDEFGHIJK"),
        ("https://youtu.be/ABCDEFGHIJK", "ABCDEFGHIJK"),
        ("https://www.youtube.com/watch?v=ABCDEFGHIJK&feature=share", "ABCDEFGHIJK"),
        ("https://www.youtube.com/embed/ABCDEFGHIJK?start=1", "ABCDEFGHIJK"),
    ],
)
def test_get_youtube_video_id_valid(url, expected):
    assert get_youtube_video_id(url) == expected


def test_get_youtube_video_id_invalid():
    assert get_youtube_video_id("https://example.com") is None
