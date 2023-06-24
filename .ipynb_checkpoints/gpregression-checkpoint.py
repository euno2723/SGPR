import logging

# ログの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# ログ出力の例
def my_function(x):
    logging.debug(f"入力値: {x}")

    # 処理...
    y = x * 2

    logging.debug(f"出力値: {y}")
    return y


result = my_function(5)
