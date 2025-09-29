# Futures Diff Utilities

此資料夾收錄與期貨資料比對或擷取相關的工具與筆記本。

## 環境需求

建議使用 Python 3.10 以上版本並安裝下列套件：

- `requests`
- `pandas`
- `matplotlib`

可以使用下列指令快速安裝：

```bash
pip install requests pandas matplotlib
```

## 工具列表

- `deribit_matured_futures.py`：呼叫 Deribit 公開 API 取得已到期的期貨合約清單、結算歷史紀錄，並對照今日日期整理出已完成結算且確定到期的期貨列表；同時為每檔合約抓取到期前指定小時數（預設 240 小時）的每小時價格資料，可選擇只保留收盤價或完整 OHLC/成交量資訊，並輸出為 CSV。此外，也能比較成熟合約與永續合約的價差、以及成熟合約與仍在交易、擁有最長歷史紀錄的期貨之間的價格走勢。

執行方式：

```bash
python futures_diff/deribit_matured_futures.py \
	--currencies BTC \
	--count 200 \
	--max-pages 10 \
	--rows 10 \
	--hours-before 240 \
	--close-only \
	--include-perp-spread \
	--plot-spread \
	--plot-reference-comparison \
	--export-dir data/hourly
```

常用參數：

- `--rows`：輸出表格列數；設成 0 可顯示全部列。
- `--show-instruments`、`--show-history`：可額外顯示原始 matured 合約清單或結算紀錄。
- `--hours-before`：抓取到期前 N 小時的每小時價格，預設 240 小時（10 天）。
- `--close-only`：只輸出 `timestamp + close` 欄位；若未提供則輸出完整 OHLCV。
- `--include-perp-spread`：同時抓取對應永續合約（例如 BTC-PERPETUAL）的每小時收盤價，計算期貨收盤價與永續收盤價的差值。
- `--plot-spread`：搭配 `--include-perp-spread` 使用，完成計算後以折線圖顯示各合約的收盤價差（需要 matplotlib）。
- `--plot-reference-comparison`：抓取仍在交易且歷史最久的期貨商品，並與每檔成熟合約在相同時間區間內的價格進行折線圖比較，同時輸出兩者價格差（matured - reference）的折線圖與對應 CSV。
- `--export-dir`：指定資料夾（自動建立）儲存各合約的 CSV 檔；檔名會依是否只含收盤價及是否包含價差附上 `_close`、`_ohlc`、`_spread` 後綴。未填時僅在終端顯示。

若無任何 matured 合約或對應價格資料，程式會顯示空集合提示。
