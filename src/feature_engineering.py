import pandas as pd
import numpy as np


def add_calendar_features(df):
    # chuỗi doanh thu bị chi phối mạnh bởi lịch: mùa vụ năm, vị trí ngày trong tháng,
    # đầu tháng / cuối tháng, khác biệt năm chẵn - năm lẻ...
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear

    # Cờ cuối tuần để bắt khác biệt hành vi mua sắm theo tuần
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cờ năm lẻ để giữ các regime kiểu promo chỉ xuất hiện năm lẻ
    df["is_odd_year"] = (df["year"] % 2 != 0).astype(int)

    # Khoảng cách tới cuối tháng.
    # Insight thấy cuối tháng / đầu tháng là vùng rất quan trọng.
    df["days_to_month_end"] = (
        (df["Date"] + pd.offsets.MonthEnd(0)) - df["Date"]
    ).dt.days

    # window 3 ngày đầu tháng và 3 ngày cuối tháng
    # dùng để bắt các spike chi tiêu theo chu kỳ tháng.
    df["is_month_start_window"] = (df["day"] <= 3).astype(int)
    df["is_month_end_window"] = (df["days_to_month_end"] <= 3).astype(int)

    # Tuần thứ mấy trong tháng, dùng để mô tả nhịp tiêu dùng thô trong tháng
    df["week_of_month"] = ((df["day"] - 1) // 7 + 1).astype(int)

    # Giai đoạn sau 2018 có dấu hiệu đổi mức nền / regime,
    # nên thêm cờ và trend riêng cho giai đoạn đó.
    df["post_2018"] = (df["Date"] >= pd.Timestamp("2019-01-01")).astype(int)
    df["post_2018_trend"] = df["post_2018"] * (
        (df["Date"] - pd.Timestamp("2019-01-01")).dt.days.clip(lower=0)
    )

    # Fourier terms giúp mô hình học mùa vụ năm mượt hơn
    # so với chỉ dùng month/day rời rạc.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    df["dayofmonth_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["dayofmonth_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    # Tầm tháng 4-6 Revenue cao
    df["is_peak_season"] = df["month"].isin([4, 5, 6]).astype(int)
    df["is_mid_year"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_year_end"] = df["month"].isin([11, 12]).astype(int)

    # Tháng 8 năm lẻ là regime cần đánh dấu riêng
    df["is_august_odd_year"] = ((df["month"] == 8) & (df["is_odd_year"] == 1)).astype(
        int
    )

    return df


def add_promo_features(df):
    # dùng các khung promo có pattern lặp lại trong lịch sử.
    m = df["month"]
    d = df["day"]
    odd = df["is_odd_year"] == 1

    df["is_spring_sale"] = (((m == 3) & (d >= 18)) | ((m == 4) & (d <= 17))).astype(int)
    df["is_mid_year_sale"] = (((m == 6) & (d >= 23)) | ((m == 7) & (d <= 22))).astype(
        int
    )
    df["is_fall_launch"] = (
        ((m == 8) & (d >= 30)) | (m == 9) | ((m == 10) & (d <= 2))
    ).astype(int)
    df["is_year_end_sale"] = (
        ((m == 11) & (d >= 18)) | (m == 12) | ((m == 1) & (d <= 2))
    ).astype(int)
    # Hai campaign chỉ nổi lên ở năm lẻ
    df["is_urban_blowout"] = (
        odd & (((m == 7) & (d >= 30)) | (m == 8) | ((m == 9) & (d <= 2)))
    ).astype(int)
    df["is_rural_special"] = (
        odd & (((m == 1) & (d >= 30)) | (m == 2) | ((m == 3) & (d <= 1)))
    ).astype(int)

    promo_cols = [
        "is_spring_sale",
        "is_mid_year_sale",
        "is_fall_launch",
        "is_year_end_sale",
        "is_urban_blowout",
        "is_rural_special",
    ]

    # Có bao nhiêu campaign đang active trong ngày đó
    df["active_promos_count"] = df[promo_cols].sum(axis=1)

    # Tương tác giữa số promo active với window đầu / cuối tháng
    # vì spike mạnh thường xảy ra khi promo chồng đúng nhịp chi tiêu.
    df["promo_x_month_end"] = df["active_promos_count"] * df["is_month_end_window"]
    df["promo_x_month_start"] = df["active_promos_count"] * df["is_month_start_window"]

    return df


def build_order_daily(orders, order_items, products):
    # Tạo thống kê giao dịch theo ngày từ orders + order_items + products
    ord_daily = (
        orders.groupby("order_date")
        .agg(order_cnt=("order_id", "nunique"))
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )

    oi = (
        order_items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
        .merge(products[["product_id", "price", "cogs"]], on="product_id", how="left")
        .copy()
    )

    # không khuyến mãi thì discount_amount là 0
    oi["discount_amount"] = oi["discount_amount"].fillna(0)
    # doanh thu theo đơn giá
    oi["order_line_value"] = oi["quantity"] * oi["unit_price"]
    # giá trị thực thu sau khuyến mãi
    oi["net_line_value"] = oi["order_line_value"] - oi["discount_amount"]
    # dòng sản phẩm có khuyến mãi hay không
    oi["promo_flag"] = (
        oi["promo_id"].notna()
        | oi["promo_id_2"].notna()
        | (oi["discount_amount"].fillna(0) > 0)
    ).astype(int)

    # mô tả product mix của lịch sử, không phải thật trong tương lai
    oi["list_line_value"] = oi["quantity"] * oi["price"]  # giá bán lẻ
    oi["static_cogs_line_value"] = oi["quantity"] * oi["cogs"]  # giá vốn tĩnh

    items_daily = (
        oi.groupby("order_date")
        .agg(
            # tổng số đvi bán ra trong ngày
            units_sold=("quantity", "sum"),
            # số dòng item trong ngày
            line_cnt=("order_id", "size"),
            # độ đa dạng product mix trong ngày
            unique_product_cnt=("product_id", "nunique"),
            # gross rev trước khuyến mãi
            order_line_value=("order_line_value", "sum"),
            # tổng giá trị sau khuyến mãi
            net_line_value=("net_line_value", "sum"),
            # tổng tiền giảm giá trong ngày
            discount_amount=("discount_amount", "sum"),
            # tỷ lệ dòng hàng có khuyến mãi
            promo_line_share=("promo_flag", "mean"),
            # tổng giá bán lẻ theo dòng hàng
            list_line_value=("list_line_value", "sum"),
            # tổng cogs theo dòng hàng
            static_cogs_line_value=("static_cogs_line_value", "sum"),
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )

    daily = (
        ord_daily.merge(items_daily, on="Date", how="outer")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fill0_cols = [
        "order_cnt",
        "units_sold",
        "line_cnt",
        "unique_product_cnt",
        "order_line_value",
        "net_line_value",
        "discount_amount",
        "promo_line_share",
        "list_line_value",
        "static_cogs_line_value",
    ]

    for c in fill0_cols:
        daily[c] = daily[c].fillna(0)

    oc = daily["order_cnt"].replace(0, np.nan)
    us = daily["units_sold"].replace(0, np.nan)
    gv = daily["order_line_value"].replace(0, np.nan)
    lv = daily["list_line_value"].replace(0, np.nan)

    # số đvi sản phẩm trung bình trên mỗi order
    daily["units_per_order"] = daily["units_sold"] / oc
    # số dòng sp trung bình trên mõi order
    daily["lines_per_order"] = daily["line_cnt"] / oc

    # giá bán trước khuyến mãi bình quân theo unit
    daily["avg_unit_price_w"] = daily["order_line_value"] / us
    # effective discount rate trong ngày
    daily["discount_rate"] = daily["discount_amount"] / gv

    # giá bán bình quân theo unit
    daily["prod_list_price_per_unit"] = daily["list_line_value"] / us
    # cogs bình quân theo unit
    daily["prod_cogs_per_unit"] = daily["static_cogs_line_value"] / us

    # Tỷ lệ giá bán trước khuyến mãi trên đơn hàng so với giá bán lẻ master.
    # Nếu < 1: order unit_price thấp hơn product price trong master.
    # Nếu > 1: order unit_price cao hơn product price trong master.
    daily["order_price_vs_list"] = daily["order_line_value"] / lv
    # biên lợi nhuận tĩnh
    daily["prod_static_margin_rate"] = (
        daily["list_line_value"] - daily["static_cogs_line_value"]
    ) / lv

    daily["month"] = daily["Date"].dt.month
    daily["day"] = daily["Date"].dt.day
    daily["dayofweek"] = daily["Date"].dt.dayofweek

    return daily


def add_order_templates(df, daily_fit):
    # Chỉ fit các template lịch sử bằng dữ liệu có sẵn đến cuối tập train của fold.
    # Trong cross-validation, daily_fit chỉ được chứa order history có Date <= train_end.
    # Mục tiêu là tránh dùng thông tin validation/tương lai khi tạo aggregate features.
    df = df.copy()
    daily_fit = daily_fit.copy()

    ratio_cols = [
        "units_per_order",
        "lines_per_order",
        "avg_unit_price_w",
        "discount_rate",
        "prod_list_price_per_unit",
        "prod_cogs_per_unit",
        "order_price_vs_list",
        "prod_static_margin_rate",
    ]

    for c in ratio_cols:
        daily_fit[c] = daily_fit[c].replace([np.inf, -np.inf], np.nan)
        med = daily_fit[c].median()
        if pd.isna(med):
            med = 0
        daily_fit[c] = daily_fit[c].fillna(med)

    tpl_md = (
        daily_fit.groupby(["month", "day"])
        .agg(
            # demand scale template
            tpl_order_cnt_md=("order_cnt", "median"),
            tpl_units_sold_md=("units_sold", "median"),
            tpl_unique_product_cnt_md=("unique_product_cnt", "median"),
            # basket behavior template
            tpl_units_per_order_md=("units_per_order", "median"),
            tpl_lines_per_order_md=("lines_per_order", "median"),
            # price / discount / product mix template
            tpl_avg_unit_price_w_md=("avg_unit_price_w", "median"),
            tpl_discount_rate_md=("discount_rate", "median"),
            tpl_prod_list_price_per_unit_md=("prod_list_price_per_unit", "median"),
            tpl_prod_cogs_per_unit_md=("prod_cogs_per_unit", "median"),
            tpl_order_price_vs_list_md=("order_price_vs_list", "median"),
            tpl_prod_static_margin_rate_md=("prod_static_margin_rate", "median"),
        )
        .reset_index()
    )

    tpl_mdow = (
        daily_fit.groupby(["month", "dayofweek"])
        .agg(
            # demand scale template
            tpl_order_cnt_mdow=("order_cnt", "median"),
            tpl_units_sold_mdow=("units_sold", "median"),
            tpl_unique_product_cnt_mdow=("unique_product_cnt", "median"),
            # price / discount / product mix template
            tpl_avg_unit_price_w_mdow=("avg_unit_price_w", "median"),
            tpl_discount_rate_mdow=("discount_rate", "median"),
            tpl_prod_list_price_per_unit_mdow=("prod_list_price_per_unit", "median"),
            tpl_prod_cogs_per_unit_mdow=("prod_cogs_per_unit", "median"),
            tpl_order_price_vs_list_mdow=("order_price_vs_list", "median"),
            tpl_prod_static_margin_rate_mdow=("prod_static_margin_rate", "median"),
        )
        .reset_index()
    )

    df = df.merge(tpl_md, on=["month", "day"], how="left")
    df = df.merge(tpl_mdow, on=["month", "dayofweek"], how="left")

    tpl_md_cols = [c for c in tpl_md.columns if c.startswith("tpl_")]
    tpl_mdow_cols = [c for c in tpl_mdow.columns if c.startswith("tpl_")]

    for c in tpl_md_cols:
        fallback = tpl_md[c].median()
        if pd.isna(fallback):
            fallback = 0
        df[c] = df[c].fillna(fallback)

    for c in tpl_mdow_cols:
        fallback = tpl_mdow[c].median()
        if pd.isna(fallback):
            fallback = 0
        df[c] = df[c].fillna(fallback)

    return df
