"""冒烟测试：包能导入，结构没坏。"""

import diy_llm


def test_version():
    assert diy_llm.__version__
