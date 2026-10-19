"""
Generator script for SpektraDarkroom Session (.sdss) document file icon.
Produces resources/sdss_icon.png and resources/sdss_icon.ico with:
1. Genuine page fold geometry with cut and transparent top-right corner;
2. Soft ambient drop shadow underneath the paper and under the folded flap;
3. Ample breathing room between the fold and the 35mm film negative strip, preserving all 8 sprocket holes;
4. High-contrast, sharp amber gold "SDSS" badge for crisp rendering at all Windows Explorer icon sizes.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def generate_sdss_icon(size: int = 256) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    scale = size / 256.0

    # 1. Dimensions matching the document silhouette
    left = int(32 * scale)
    right = int(224 * scale)
    top = int(20 * scale)
    bottom = int(236 * scale)
    r = int(16 * scale)
    fold = int(46 * scale)

    # Fold vertices
    fold_top_x = right - fold
    fold_top_y = top
    fold_right_x = right
    fold_right_y = top + fold
    fold_crease_x = right - fold
    fold_crease_y = top + fold

    # 2. Build Paper Silhouette Mask (4x Super Sampling for crisp anti-aliasing)
    ss = 4
    w_ss = size * ss
    h_ss = size * ss
    mask_ss = Image.new("L", (w_ss, h_ss), 0)
    draw_m = ImageDraw.Draw(mask_ss)

    l_s = left * ss
    r_s = right * ss
    t_s = top * ss
    b_s = bottom * ss
    rad_s = r * ss
    f_s = fold * ss

    # Base rounded rectangle
    draw_m.rounded_rectangle([l_s, t_s, r_s, b_s], radius=rad_s, fill=255)
    # Cut off top right corner along diagonal (strictly镂空透背)
    cut_poly = [(r_s - f_s, t_s - 2), (r_s + 4, t_s - 2), (r_s + 4, t_s + f_s)]
    draw_m.polygon(cut_poly, fill=0)

    mask = mask_ss.resize((size, size), Image.Resampling.LANCZOS)

    # 3. Ambient Drop Shadow under document
    shadow_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(int(8 * scale)))
    shadow_img.paste((0, 0, 0, 110), (0, int(6 * scale)), mask=shadow_mask)
    img = Image.alpha_composite(img, shadow_img)

    # 4. Draw Document Body (Obsidian slate #181a21 with clean 1px highlight border)
    paper_layer = Image.new("RGBA", (size, size), (24, 26, 33, 255))
    paper_layer.putalpha(mask)

    # Subtle clean border
    border_mask_ss = Image.new("L", (w_ss, h_ss), 0)
    draw_bm = ImageDraw.Draw(border_mask_ss)
    draw_bm.rounded_rectangle([l_s, t_s, r_s, b_s], radius=rad_s, outline=255, width=int(2 * ss * scale))
    draw_bm.polygon(cut_poly, fill=0)
    draw_bm.line([(r_s - f_s, t_s), (r_s, t_s + f_s)], fill=255, width=int(2 * ss * scale))
    border_mask = border_mask_ss.resize((size, size), Image.Resampling.LANCZOS)

    border_layer = Image.new("RGBA", (size, size), (46, 50, 64, 255))
    border_layer.putalpha(border_mask)
    paper_layer = Image.alpha_composite(paper_layer, border_layer)
    img = Image.alpha_composite(img, paper_layer)

    # 5. Folded Flap Drop Shadow (casts on paper downwards and left)
    flap_shadow_ss = Image.new("L", (w_ss, h_ss), 0)
    draw_fss = ImageDraw.Draw(flap_shadow_ss)
    ft_x = fold_top_x * ss
    ft_y = fold_top_y * ss
    fr_x = fold_right_x * ss
    fr_y = fold_right_y * ss
    fc_x = fold_crease_x * ss
    fc_y = fold_crease_y * ss
    draw_fss.polygon([(ft_x, ft_y), (fr_x, fr_y), (fc_x, fc_y)], fill=255)
    flap_shadow = flap_shadow_ss.resize((size, size), Image.Resampling.LANCZOS)
    flap_shadow = flap_shadow.filter(ImageFilter.GaussianBlur(int(4 * scale)))

    flap_sh_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    flap_sh_layer.paste((0, 0, 0, 140), (int(-2 * scale), int(4 * scale)), mask=flap_shadow)
    flap_sh_layer.putalpha(Image.composite(flap_sh_layer.getchannel('A'), Image.new("L", (size, size), 0), mask))
    img = Image.alpha_composite(img, flap_sh_layer)

    # 6. Folded Flap Surface
    flap_ss = Image.new("RGBA", (w_ss, h_ss), (0, 0, 0, 0))
    draw_flap = ImageDraw.Draw(flap_ss)
    draw_flap.polygon([(ft_x, ft_y), (fr_x, fr_y), (fc_x, fc_y)], fill=(44, 48, 62, 255), outline=(68, 74, 96, 255))
    flap_img = flap_ss.resize((size, size), Image.Resampling.LANCZOS)
    img = Image.alpha_composite(img, flap_img)

    # 7. Center 35mm Film Negative Strip
    # Positioned with comfortable breathing room below fold (y: 72..156)
    film_x = int(58 * scale)
    film_y = int(72 * scale)
    film_w = int(140 * scale)
    film_h = int(84 * scale)
    film_r = int(6 * scale)

    film_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_film = ImageDraw.Draw(film_layer)
    draw_film.rounded_rectangle(
        [film_x, film_y, film_x + film_w, film_y + film_h],
        radius=film_r,
        fill=(11, 12, 15, 255),
        outline=(34, 38, 48, 255),
        width=int(1 * scale)
    )

    # Film sprocket holes: 8 evenly spaced rounded holes
    num_sprockets = 8
    sp_w = int(10 * scale)
    sp_h = int(9 * scale)
    sp_r = int(2 * scale)
    sp_margin = int(8 * scale)
    available_w = film_w - 2 * sp_margin - sp_w
    sp_step = available_w / (num_sprockets - 1)

    for i in range(num_sprockets):
        sx = int(film_x + sp_margin + i * sp_step)
        # Top sprocket
        draw_film.rounded_rectangle(
            [sx, film_y + int(4 * scale), sx + sp_w, film_y + int(4 * scale) + sp_h],
            radius=sp_r,
            fill=(24, 26, 33, 255),
            outline=(18, 19, 24, 255),
            width=max(1, int(1 * scale))
        )
        # Bottom sprocket
        draw_film.rounded_rectangle(
            [sx, film_y + film_h - int(4 * scale) - sp_h, sx + sp_w, film_y + film_h - int(4 * scale)],
            radius=sp_r,
            fill=(24, 26, 33, 255),
            outline=(18, 19, 24, 255),
            width=max(1, int(1 * scale))
        )

    # Center frame viewport: Amber Gold frame (x: 90..166, y: 86..142)
    frame_x = int(90 * scale)
    frame_y = int(86 * scale)
    frame_w = int(76 * scale)
    frame_h = int(56 * scale)

    draw_film.rectangle(
        [frame_x, frame_y, frame_x + frame_w, frame_y + frame_h],
        outline=(245, 158, 11, 255),
        width=max(1, int(2 * scale))
    )

    # Optical center viewfinder circle
    c_x = frame_x + frame_w // 2
    c_y = frame_y + frame_h // 2
    c_r = int(9 * scale)
    draw_film.ellipse(
        [c_x - c_r, c_y - c_r, c_x + c_r, c_y + c_r],
        outline=(245, 158, 11, 255),
        width=max(1, int(2 * scale))
    )

    img = Image.alpha_composite(img, film_layer)

    # 8. Bottom SDSS Amber Gold Badge (x: 50..206, y: 172..218)
    badge_x = int(50 * scale)
    badge_y = int(172 * scale)
    badge_w = int(156 * scale)
    badge_h = int(46 * scale)
    badge_r = int(10 * scale)

    # Subtle drop shadow for badge
    b_sh_mask = Image.new("L", (size, size), 0)
    draw_bsh = ImageDraw.Draw(b_sh_mask)
    draw_bsh.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], radius=badge_r, fill=255)
    b_sh_mask = b_sh_mask.filter(ImageFilter.GaussianBlur(int(3 * scale)))
    badge_sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    badge_sh.paste((0, 0, 0, 90), (0, int(2 * scale)), mask=b_sh_mask)
    img = Image.alpha_composite(img, badge_sh)

    badge_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_badge = ImageDraw.Draw(badge_layer)
    draw_badge.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=badge_r,
        fill=(245, 158, 11, 255)
    )

    # Bold "SDSS" text
    font_bold = "C:/Windows/Fonts/segoeuib.ttf"
    if not os.path.exists(font_bold):
        font_bold = "C:/Windows/Fonts/arialbd.ttf"
    font = ImageFont.truetype(font_bold, int(30 * scale))
    tb = font.getbbox("SDSS")
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    tx = badge_x + (badge_w - tw) // 2 - tb[0]
    ty = badge_y + (badge_h - th) // 2 - tb[1]

    draw_badge.text((tx, ty), "SDSS", fill=(17, 18, 22, 255), font=font)
    img = Image.alpha_composite(img, badge_layer)

    return img


def update_all_sdss_icons():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res_dir = os.path.join(base_dir, "resources")
    os.makedirs(res_dir, exist_ok=True)

    png_path = os.path.join(res_dir, "sdss_icon.png")
    ico_path = os.path.join(res_dir, "sdss_icon.ico")

    # 1. Generate master 256x256 PNG
    icon_256 = generate_sdss_icon(256)
    icon_256.save(png_path, "PNG")
    print(f"Successfully updated {png_path} (256x256 RGBA)")

    # 2. Generate multi-resolution Windows ICO
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icon_256.save(ico_path, format="ICO", sizes=ico_sizes)
    print(f"Successfully updated {ico_path} with sizes: {ico_sizes}")


if __name__ == "__main__":
    update_all_sdss_icons()
