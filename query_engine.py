def get_performance_data(conn, days: int):
    if days <= 0:
        raise ValueError("`days` must be greater than 0.")

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COALESCE(AVG(discount), 0) AS avg_discount,
                COALESCE(SUM(profit), 0) AS total_profit,
                COALESCE(AVG(margin), 0) AS avg_margin,
                COUNT(*) AS transaction_count
            FROM transactions
            WHERE timestamp >= NOW() - (%s * INTERVAL '1 day')
            """,
            (days,),
        )
        row = cursor.fetchone()

    return {
        "days": days,
        "avg_discount": float(row[0]),
        "total_profit": float(row[1]),
        "avg_margin": float(row[2]),
        "transaction_count": int(row[3]),
    }


def get_business_data_profile(conn, sample_limit: int = 8) -> dict:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*) AS transaction_count,
                MIN(timestamp) AS min_timestamp,
                MAX(timestamp) AS max_timestamp,
                COUNT(DISTINCT customer_id) AS customer_count,
                COUNT(DISTINCT item_name) AS item_count
            FROM transactions
            """
        )
        overview = cursor.fetchone()

        cursor.execute(
            """
            SELECT item_name, COUNT(*) AS frequency
            FROM transactions
            WHERE item_name IS NOT NULL AND item_name <> ''
            GROUP BY item_name
            ORDER BY frequency DESC, item_name ASC
            LIMIT %s
            """,
            (sample_limit,),
        )
        top_items = [{"item_name": row[0], "frequency": int(row[1])} for row in cursor.fetchall()]

        cursor.execute(
            """
            SELECT customer_id, COUNT(*) AS frequency
            FROM transactions
            WHERE customer_id IS NOT NULL AND customer_id <> ''
            GROUP BY customer_id
            ORDER BY frequency DESC, customer_id ASC
            LIMIT %s
            """,
            (sample_limit,),
        )
        top_customers = [{"customer_id": row[0], "frequency": int(row[1])} for row in cursor.fetchall()]

    return {
        "transaction_count": int(overview[0] or 0),
        "min_timestamp": overview[1],
        "max_timestamp": overview[2],
        "customer_count": int(overview[3] or 0),
        "item_count": int(overview[4] or 0),
        "top_items": top_items,
        "top_customers": top_customers,
        "available_dimensions": [
            "customer_id",
            "item_name",
            "item_id",
            "price",
            "discount",
            "gst",
            "total_price",
            "profit",
            "margin",
            "timestamp",
        ],
    }
