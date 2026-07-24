# -*- coding: utf-8 -*-
from pathlib import Path
import shutil

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


SRC = Path(r"C:\Users\zx\Desktop\模板.pptx")
OUT = Path("outputs/模板_前两部分修改版.pptx")

COL = {
    "ink": "172033",
    "muted": "56657A",
    "line": "D7DFEA",
    "bg": "F7FAFC",
    "white": "FFFFFF",
    "blue": "1F5EFF",
    "teal": "00A99D",
    "cyan": "1DA1F2",
    "orange": "F59E0B",
    "red": "D64545",
    "green": "15A66A",
    "violet": "7C3AED",
    "dark": "13223A",
}


def rgb(hex_color):
    h = hex_color.replace("#", "")
    return RGBColor(int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


def fill(shape, color, transparency=0):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(color)
    if transparency:
        shape.fill.transparency = transparency


def line(shape, color=None, width=0.75):
    if color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = rgb(color)
        shape.line.width = Pt(width)


def text(slide, value, x, y, w, h, size=16, color="172033", bold=False,
         align="left", valign="top", font="Microsoft YaHei", margin=0.04):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = {
        "top": MSO_ANCHOR.TOP,
        "middle": MSO_ANCHOR.MIDDLE,
        "bottom": MSO_ANCHOR.BOTTOM,
    }[valign]
    p = tf.paragraphs[0]
    p.alignment = {
        "left": PP_ALIGN.LEFT,
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
    }[align]
    run = p.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def multiline(slide, values, x, y, w, h, size=12.5, color="172033", gap=2, font="Microsoft YaHei"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for i, value in enumerate(values):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = value
        p.font.name = font
        p.font.size = Pt(size)
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(gap)
    return box


def cover(slide):
    # Covers old body content but keeps the template header/sidebar decoration.
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.15), Inches(1.55), Inches(17.7), Inches(8.85))
    fill(r, COL["bg"])
    line(r, None)
    return r


def clear_top_level_text(slide):
    # Keep grouped template titles, remove old body text boxes before adding revised content.
    for sh in slide.shapes:
        if sh.shape_type != MSO_SHAPE_TYPE.GROUP and hasattr(sh, "text_frame"):
            sh.text_frame.clear()


def title(slide, value, subtitle=None):
    text(slide, value, 1.45, 1.78, 15.8, 0.52, 25, COL["ink"], True)
    if subtitle:
        text(slide, subtitle, 1.48, 2.33, 15.6, 0.34, 12.5, COL["muted"])


def card(slide, x, y, w, h, head, body=None, accent="1F5EFF", head_size=15, body_size=11.5):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    fill(s, COL["white"])
    line(s, COL["line"], 0.8)
    b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    fill(b, accent)
    line(b, None)
    text(slide, head, x + 0.2, y + 0.14, w - 0.35, 0.34, head_size, COL["ink"], True)
    if body:
        if isinstance(body, list):
            multiline(slide, body, x + 0.2, y + 0.55, w - 0.35, h - 0.65, body_size, COL["muted"])
        else:
            text(slide, body, x + 0.2, y + 0.57, w - 0.35, h - 0.65, body_size, COL["muted"])
    return s


def badge(slide, value, x, y, w, color="1F5EFF", size=10.5):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.34))
    fill(s, color)
    line(s, None)
    text(slide, value, x + 0.02, y + 0.06, w - 0.04, 0.16, size, COL["white"], True, "center")
    return s


def arrow(slide, x1, y1, x2, y2, color="56657A"):
    a = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    a.line.color.rgb = rgb(color)
    a.line.width = Pt(1.5)
    a.line.end_arrowhead = True
    return a


def source(slide, value):
    text(slide, value, 1.45, 10.25, 16.8, 0.22, 7.8, COL["muted"])


def replace_group_title(slide, old, new):
    def visit(shapes):
        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                visit(sh.shapes)
            elif hasattr(sh, "text") and sh.text.strip() == old:
                sh.text = new
    visit(slide.shapes)


def add_table(slide, headers, rows, x, y, col_w, row_h=0.62, header_color="13223A",
              font_size=9.4, header_size=10.5):
    for j, h in enumerate(headers):
        xx = x + sum(col_w[:j])
        s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xx), Inches(y), Inches(col_w[j]), Inches(row_h))
        fill(s, header_color)
        line(s, COL["white"], 0.4)
        text(slide, h, xx + 0.06, y + 0.18, col_w[j] - 0.12, 0.16, header_size, COL["white"], True, "center")
    for i, row in enumerate(rows):
        yy = y + row_h * (i + 1)
        for j, cell in enumerate(row):
            xx = x + sum(col_w[:j])
            s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(xx), Inches(yy), Inches(col_w[j]), Inches(row_h))
            fill(s, "FFFFFF" if i % 2 == 0 else "EEF4FA")
            line(s, COL["line"], 0.45)
            text(slide, cell, xx + 0.07, yy + 0.09, col_w[j] - 0.14, row_h - 0.12, font_size, COL["ink"])


def slide4_system(slide):
    replace_group_title(slide, "整体进展", "系统优化与部署")
    clear_top_level_text(slide)
    cover(slide)
    title(slide, "系统优化与部署：补充“部署可交付”的内容", "第一页建议从单纯架构对比，补成“部署对象、部署约束、可验收结果”三块。")
    card(slide, 1.55, 3.00, 5.1, 3.15, "部署对象", [
        "FastAPI 后端：统一分析入口与结果返回",
        "Streamlit 面板：上传、展示概率和证据链",
        "SQLite / 指纹库：保存 hash、pHash、Top-K 来源",
        "Linux 部署文档：按“先跑通，再增强”组织",
    ], COL["blue"], 16, 11.5)
    card(slide, 7.30, 3.00, 5.1, 3.15, "部署约束", [
        "API Key 不写入配置文件，改用环境变量",
        "HF_HOME / 数据集缓存固定到项目目录",
        "c2patool 使用 Linux 可执行文件路径",
        "不使用 reload，避免模型下载触发重启",
    ], COL["teal"], 16, 11.5)
    card(slide, 13.05, 3.00, 5.1, 3.15, "验收结果", [
        "健康检查：/health、/providers 可访问",
        "文本/图片/音频/视频上传链路可跑通",
        "报告输出：AI 概率 + 来源候选 + 证据说明",
        "增强项：C2PA、水印、来源归因逐步接入",
    ], COL["orange"], 16, 11.5)
    badge(slide, "可讲重点", 1.55, 7.05, 1.25, COL["dark"])
    text(slide, "部署不是只写启动命令，而是把运行依赖、模型缓存、安全配置、证据链输出和后续增强项都固定下来。", 3.00, 7.05, 14.8, 0.36, 15, COL["ink"], True)
    # Mini delivery ladder
    steps = [("基础运行", "API + Dashboard"), ("模型准备", "HF 缓存 + 分类器"), ("溯源增强", "C2PA + 水印"), ("报告验收", "证据链 + 限制说明")]
    for i, (h, b) in enumerate(steps):
        x = 2.0 + i * 4.0
        badge(slide, h, x, 8.42, 1.15, [COL["blue"], COL["teal"], COL["orange"], COL["violet"]][i], 9)
        text(slide, b, x - 0.45, 8.88, 2.05, 0.26, 10.5, COL["muted"], True, "center")
        if i < 3:
            arrow(slide, x + 1.3, 8.58, x + 3.35, 8.58, COL["muted"])
    source(slide, "对应文档：docs/LINUX_DEPLOYMENT.md；对应系统：FastAPI / Streamlit / SQLite / provenance pipeline")


def slide5_generation(slide):
    replace_group_title(slide, "溯源数据集", "生成样本规范")
    clear_top_level_text(slide)
    cover(slide)
    title(slide, "多模型、多领域样本生成规范", "固定提示词模板和输出格式，便于后续检测评测、来源归因训练和误报率分析。")
    headers = ["模态", "模型来源", "领域/任务", "固定提示词或要求", "输出与标注字段"]
    rows = [
        ("文本", "DeepSeek / Kimi / Qwen / GPT / Gemini / MiniMax", "新闻、学术、教育、法律、医疗、技术文档", "同一主题统一生成：摘要、扩写、润色、改写、问答；限制字数、语气、语言", "JSONL：text, source_model, domain, prompt_type, language, operation"),
        ("图片", "GPT Image / Midjourney / FLUX / SDXL / Firefly", "人物、商品、新闻现场、证件/海报、自然场景", "同一 prompt 固定风格、分辨率、构图；额外生成截图、裁剪、压缩版本", "目录：model/domain/prompt_id；记录 prompt、seed、size、transform"),
        ("对照", "人工文本 / 真实图片 / 公开数据", "与 AI 样本领域一致", "人工样本不混入模型提示词；真实图保留原始来源与处理历史", "label=human/real，用于估计误报率"),
        ("评测", "同模型不同提示词 + 不同模型同提示词", "跨模型、跨领域、跨操作", "统一 prompt_id，让同一任务在多个模型间可横向比较", "可计算 domain shift、model shift、attack shift"),
    ]
    add_table(slide, headers, rows, 1.35, 2.88, [1.25, 4.15, 3.25, 4.35, 4.35], 1.08, font_size=8.6, header_size=9.5)
    badge(slide, "示例 JSONL", 1.45, 7.75, 1.45, COL["teal"], 9.2)
    code = '{"text":"...", "label":"ai", "source_model":"DeepSeek", "domain":"academic", "prompt_type":"rewrite", "operation":"polish", "language":"zh"}'
    text(slide, code, 3.05, 7.75, 14.2, 0.42, 10.2, COL["ink"], False, font="Consolas")
    badge(slide, "图片目录", 1.45, 8.65, 1.45, COL["orange"], 9.2)
    text(slide, "data/attribution/image_samples/<model>/<domain>/<prompt_id>_<transform>.png", 3.05, 8.65, 14.2, 0.34, 10.2, COL["ink"], False, font="Consolas")
    source(slide, "建议：文本和图片都保存 prompt_id，保证“同提示词跨模型”和“同模型跨领域”可复现实验。")


def slide7_text(slide):
    clear_top_level_text(slide)
    cover(slide)
    title(slide, "文本领域：不足与可用点", "2026 年文本检测重点不是“是否能检”，而是真实任务、多语言、跨模型和改写攻击下是否稳定。")
    headers = ["2026 工作", "暴露的不足", "本系统可用点", "链接"]
    rows = [
        ("DetectRL-X", "现有检测器在多语言、商业场景和润色/扩写/缩写/改写后稳定性不足。", "样本字段加入 language、domain、operation、attack_type；按语言分别评估。", "arxiv.org/abs/2605.15518"),
        ("TSM-Bench", "真实 Wikipedia 编辑任务比泛泛生成更像人写，检测准确率下降 10-40%。", "生成样本不能只写文章，要覆盖摘要、补写、改写、润色等真实任务。", "arxiv.org/abs/2605.31113"),
        ("Detecting the Machine", "域内分数高不代表跨域可用；LLM-as-detector 也存在身份偏置。", "报告中区分训练分布内/外，并保留 stylometry 与 PPL 作为辅助特征。", "arxiv.org/abs/2603.17522"),
        ("LLM-specific FT", "按来源模型/模型族微调效果好，但维护成本随模型更新上升。", "用于 Top-K 来源归因：DeepSeek、Kimi、Qwen、GPT、Gemini、MiniMax。", "arxiv.org/abs/2601.20006"),
    ]
    add_table(slide, headers, rows, 1.28, 2.88, [2.55, 4.55, 5.25, 4.40], 1.16, font_size=8.5, header_size=9.5)
    badge(slide, "结论", 1.45, 8.45, 1.05, COL["dark"], 9.5)
    text(slide, "文本检测页应强调“不足 + 可用点”：不足是跨域、跨模型、改写攻击；可用点是规范样本字段、Top-K 来源归因、报告中说明限制。", 2.75, 8.43, 15.1, 0.42, 13.5, COL["ink"], True)
    source(slide, "来源：DetectRL-X / TSM-Bench / Detecting the Machine / LLM-specific fine-tuning（均为 2026 arXiv）")


def slide8_image(slide):
    clear_top_level_text(slide)
    cover(slide)
    title(slide, "图像领域：真实传播后检测难度上升", "裁剪、拉伸、压缩、截图、多次转发会破坏元数据和水印，检测只能更多依赖像素/频域/风格特征。")
    # propagation flow
    nodes = [
        ("原始 AI 图像", "可能带 C2PA / 水印"),
        ("传播变换", "裁剪、拉伸、截图、压缩"),
        ("平台转发", "CDN 重编码、metadata 丢失"),
        ("检测侧", "学习像素、频域、模型风格"),
    ]
    colors = [COL["blue"], COL["orange"], COL["red"], COL["teal"]]
    for i, (h, b) in enumerate(nodes):
        x = 1.45 + i * 4.25
        card(slide, x, 2.80, 3.25, 1.25, h, b, colors[i], 13.5, 9.5)
        if i < 3:
            arrow(slide, x + 3.25, 3.42, x + 4.00, 3.42, COL["muted"])
    headers = ["2026 依据", "关键发现", "本项目可用点"]
    rows = [
        ("NTIRE 2026", "108,750 真实图、185,750 AI 图、42 个生成器、36 种真实变换，评估裁剪/缩放/压缩/模糊后鲁棒性。", "图片样本库加入 transform 字段，保存原图与变换后版本。"),
        ("Out-of-box Benchmark", "16 类方法、23 个预训练 detector、291 个生成器；没有通用赢家，新商业模型常使检测器失效。", "不要绑定单一 detector，使用 provider ensemble 并持续更新。"),
        ("GPT-Image-2 Twitter", "X/Twitter 上传后 C2PA 内容凭证被系统性剥离，社交媒体来源图像难做加密溯源。", "C2PA 有则用，无则转向 pHash、CLIP embedding、像素风格归因。"),
    ]
    add_table(slide, headers, rows, 1.32, 5.05, [3.05, 7.3, 6.25], 0.98, font_size=8.5, header_size=9.5)
    source(slide, "链接：arxiv.org/abs/2604.11487；arxiv.org/abs/2602.07814；arxiv.org/abs/2604.25370")


def slide9_identity_samples(slide):
    clear_top_level_text(slide)
    cover(slide)
    title(slide, "多层身份识别与样本扩展方案", "溯源不依赖单一证据：内容凭证、水印、被动检测、来源归因和样本库需要同时存在。")
    headers = ["层级", "方案", "描述", "链接 / 可用点"]
    rows = [
        ("双层标识", "C2PA + SynthID", "C2PA 记录生成模型、时间、编辑历史等签名元数据；SynthID 作为更耐传播的不可见水印信号，两者互补。", "C2PA 负责可解释来源，SynthID 负责截图/转换后的残留信号。"),
        ("生产级水印", "SynthID-Text / Vaporizer / PRNG", "SynthID-Text 是生产级文本水印；Vaporizer 表明翻译、同义替换、神经改写可削弱水印；Blind PRNG Hijacking 提醒随机源供应链也会影响水印安全。", "2603.03410 / 2605.07481 / 2605.28632"),
    ]
    add_table(slide, headers, rows, 1.30, 2.82, [2.0, 3.0, 7.35, 4.4], 1.18, font_size=8.45, header_size=9.4)
    badge(slide, "样本扩展", 1.42, 5.85, 1.35, COL["teal"], 9.5)
    sample_rows = [
        ("文本", "DeepSeek/Kimi/Qwen/GPT/Gemini/MiniMax", "领域 + 任务 + 改写操作 + 语言 + 来源模型"),
        ("图片", "GPT Image/Midjourney/FLUX/SDXL/Firefly", "prompt_id + 模型 + 风格 + 裁剪/压缩/截图变换"),
        ("音视频", "TTS/音乐/视频生成/换脸/口型驱动", "片段级证据、帧级可疑区域、水印命中状态"),
    ]
    add_table(slide, ["模态", "候选来源", "必须保存字段"], sample_rows, 1.30, 6.32, [2.0, 6.0, 8.75], 0.74, font_size=8.8, header_size=9.3)
    source(slide, "结论：C2PA / 水印都不是万能答案，最终要靠“证据链 + 样本库 + 来源归因”共同支撑。")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, OUT)
    prs = Presentation(str(OUT))
    slide4_system(prs.slides[3])
    slide5_generation(prs.slides[4])
    slide7_text(prs.slides[6])
    slide8_image(prs.slides[7])
    slide9_identity_samples(prs.slides[8])
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
