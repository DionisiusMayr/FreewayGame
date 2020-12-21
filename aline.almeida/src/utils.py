def convert_score(hex_score: int) -> int:
    """Convert the score from the hex represation used in memory to base 10."""	
    return (hex_score // 16) * 10 + (hex_score % 16) 