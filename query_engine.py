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
