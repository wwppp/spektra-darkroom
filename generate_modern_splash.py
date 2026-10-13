import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_WIDTH = 1280
BASE_HEIGHT = 720


def create_splash(width: int = BASE_WIDTH, height: int = BASE_HEIGHT, out_path: str = None) -> Image.Image:
    """
    生成现代化物理胶片暗房启动画面底图。
    严格基于经过验证并被采纳的 candidate_splash_v2 原版美学：
    1. 纯正黑曜石 (13, 14, 18) 基底与右上角焦糖安全灯光学漫射；
    2. 顶部 35mm 负片齿孔细节与官方发光暗房胶卷图标；
    3. Segoe UI 常规细体典雅排版（杜绝笨重粗体与锯齿）；
    4. 适度优化副标题、特性标语与右下角标字号，显著提高高分屏清晰度；
    5. 底图完全解耦版本号（预留留白由 Qt 运行时动态合成版本号徽章）；
    6. 强制 100% 绝对实心不透明度 (putalpha 255)，彻底杜绝桌面穿透重影。
    """
    scale = width / float(BASE_WIDTH)
    
    # 1. Base dark obsidian canvas (13, 14, 18)
    img = Image.new("RGBA", (width, height), (13, 14, 18, 255))
    
    # 2. Authentic Darkroom Safelight Caramel Glow (Top-Right organic radial diffusion)
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    center_x = int(1040 * scale)
    center_y = int(140 * scale)
    max_radius = int(680 * scale)
    step = max(4, int(8 * scale))
    
    for r in range(max_radius, 0, -step):
        factor = 1.0 - (r / max_radius) ** 1.4
        alpha = int(72 * factor)
        glow_draw.ellipse(
            [center_x - r, center_y - int(r * 0.75), center_x + r, center_y + int(r * 0.75)],
            fill=(195, 95, 25, alpha)
        )
    glow = glow.filter(ImageFilter.GaussianBlur(int(50 * scale)))
    img = Image.alpha_composite(img, glow)
    
    draw = ImageDraw.Draw(img)
    
    # 3. Top 35mm Sprocket Holes
    hole_w = int(17 * scale)
    hole_h = int(27 * scale)
    hole_top = int(21 * scale)
    hole_r = max(2, int(4 * scale))
    start_x = int(36 * scale)
    spacing = int(48 * scale)
    num_holes = int((width - start_x) / spacing)
    
    for i in range(num_holes):
        hx = start_x + i * spacing
        draw.rounded_rectangle([hx, hole_top, hx + hole_w, hole_top + hole_h], radius=hole_r, fill=(8, 9, 12, 255))
        draw.rounded_rectangle(
            [hx, hole_top, hx + hole_w, hole_top + hole_h],
            radius=hole_r,
            outline=(42, 46, 58, 255),
            width=max(1, int(1 * scale))
        )
        
    # 4. Official App Icon
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, "resources", "app_icon.png")
    icon_size = int(176 * scale)
    icon_x = int(108 * scale)
    icon_y = int(236 * scale)
    
    if os.path.exists(icon_path):
        app_icon = Image.open(icon_path).convert("RGBA")
        app_icon = app_icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        
        icon_shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(icon_shadow)
        shadow_draw.rounded_rectangle(
            [icon_x + int(6 * scale), icon_y + int(12 * scale), icon_x + icon_size + int(6 * scale), icon_y + icon_size + int(12 * scale)],
            radius=int(40 * scale),
            fill=(0, 0, 0, 160)
        )
        icon_shadow = icon_shadow.filter(ImageFilter.GaussianBlur(int(16 * scale)))
        img = Image.alpha_composite(img, icon_shadow)
        icon_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        icon_layer.paste(app_icon, (icon_x, icon_y))
        img = Image.alpha_composite(img, icon_layer)
        
    draw = ImageDraw.Draw(img)
    
    # 5. Fonts setup - Clean Segoe UI Regular / Bold
    font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"
    font_regular_path = "C:/Windows/Fonts/segoeui.ttf"
    
    title_size = int(52 * scale)
    sub_size = int(24 * scale)       # 优雅适度放大至 24px，保持 Regular 纤细体
    feature_size = int(18 * scale)   # 优雅适度放大至 18px，保持 Regular 纤细体
    bar_size = int(17 * scale)       # 优雅适度放大至 17px
    
    f_title = ImageFont.truetype(font_bold_path, title_size)
    f_sub = ImageFont.truetype(font_regular_path, sub_size)
    f_feature = ImageFont.truetype(font_regular_path, feature_size)
    f_bar = ImageFont.truetype(font_regular_path, bar_size)
    
    text_x = icon_x + icon_size + int(40 * scale)
    text_y = icon_y + int(16 * scale)
    
    # SPEKTRA in Signature Amber Gold (#f59e0b)
    spektra_text = "SPEKTRA "
    draw.text((text_x, text_y), spektra_text, fill=(245, 158, 11, 255), font=f_title)
    spektra_bbox = f_title.getbbox(spektra_text)
    spektra_w = spektra_bbox[2] - spektra_bbox[0]
    
    # DARKROOM in Crisp White (#ffffff)
    darkroom_text = "DARKROOM"
    darkroom_x = text_x + spektra_w
    draw.text((darkroom_x, text_y), darkroom_text, fill=(255, 255, 255, 255), font=f_title)
    
    # 注：底图不写死版本号，右侧预留空间由运行时根据当前版本号动态绘制全圆角胶囊药丸！
    
    # Subtitle: Vulkan & GPU Accelerated Film Emulation Workstation (#d6dee8)
    sub_y = text_y + int(68 * scale)
    draw.text(
        (text_x, sub_y),
        "Vulkan & GPU Accelerated Film Emulation Workstation",
        fill=(214, 222, 232, 255),
        font=f_sub
    )
    
    # Feature Tagline (#a0aec0)
    feature_y = sub_y + int(36 * scale)
    draw.text(
        (text_x, feature_y),
        "Physical Enlarger Simulation   •   Spectral Chemistry   •   Color Science Engine",
        fill=(160, 174, 192, 255),
        font=f_feature
    )
    
    # 6. Bottom Status Bar (Height = 68px * scale)
    bar_h = int(68 * scale)
    bar_top = height - bar_h
    draw.rectangle([0, bar_top, width, height], fill=(8, 9, 12, 255))
    draw.line([0, bar_top, width, bar_top], fill=(28, 30, 38, 255), width=max(1, int(1 * scale)))
    
    # Right tagline: Designed for Analog Craft & Precision Color (#a0aec0)
    tagline_text = "Designed for Analog Craft & Precision Color"
    tag_bbox = f_bar.getbbox(tagline_text)
    tag_w = tag_bbox[2] - tag_bbox[0]
    tag_h = tag_bbox[3] - tag_bbox[1]
    tag_x = width - tag_w - int(48 * scale)
    tag_y = bar_top + (bar_h - tag_h) // 2 - tag_bbox[1]
    draw.text((tag_x, tag_y), tagline_text, fill=(160, 174, 192, 255), font=f_bar)
    
    # 强制 100% 实心绝对不透明度
    img.putalpha(255)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img.save(out_path, "PNG")
        print(f"Successfully generated splash screen [{width}x{height}] at {out_path}")
        
    return img


def generate_all_splash_screens():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "resources")
    
    specs = [
        (1280, 720, os.path.join(res_dir, "splash.png")),
        (1920, 1080, os.path.join(res_dir, "splash@1.5x.png")),
        (2560, 1440, os.path.join(res_dir, "splash@2x.png")),
    ]
    
    for w, h, path in specs:
        create_splash(width=w, height=h, out_path=path)


if __name__ == "__main__":
    generate_all_splash_screens()
