import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import get_all_symbols
import warnings
import requests
from datetime import datetime
import time
import os
import pytz
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================
# TELEGRAM AYARLARI
# ============================
bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '8035211094:AAEqHt4ZosBJsuT1FxdCcTR9p9uJ1O073zY')
bot_chatID = os.getenv('TELEGRAM_CHAT_ID', '-1002715468798')

# Türkiye Saat Dilimi
TIMEZONE = pytz.timezone('Europe/Istanbul')

# ============================
# SUNUCU AYARLARI (SABİT)
# ============================
SELECTED_INTERVAL = Interval.in_4_hour
SELECTED_BARS = 200
INTERVAL_NAME = "4 Saat"
SCAN_INTERVAL_SECONDS = 1800  # 30 dakika

print(f"✅ Telegram Bot Token: {'*'*20}{bot_token[-10:]}")
print(f"✅ Telegram Chat ID: {bot_chatID}")
print(f"✅ Interval: {INTERVAL_NAME}")
print(f"✅ Tarama sıklığı: Her 30 dakikada bir")


def get_current_time():
    return datetime.now(TIMEZONE).strftime('%d.%m.%Y %H:%M:%S')


def mesaj_at(bot_message, silent=False):
    if not bot_token or not bot_chatID:
        print("⚠️ Telegram ayarları yapılmamış!")
        return None

    api_url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    params = {
        'chat_id': bot_chatID,
        'text': bot_message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }

    try:
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200 and not silent:
            print(f"📤 Telegram mesajı gönderildi")
        return response.json()
    except Exception as e:
        if not silent:
            print(f"⚠️ Telegram hatası: {e}")
        return None


def clean_data(data):
    try:
        if data is None or len(data) == 0:
            return None

        data = data.reset_index()

        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])

        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in data.columns:
                return None
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data = data.dropna(subset=required_columns)
        data = data[
            (data['close'] > 0) & (data['open'] > 0) &
            (data['high'] > 0) & (data['low'] > 0) &
            (data['volume'] >= 0)
        ]
        data = data[
            (data['high'] >= data['low']) &
            (data['high'] >= data['close']) &
            (data['high'] >= data['open']) &
            (data['low'] <= data['close']) &
            (data['low'] <= data['open'])
        ]

        return data.reset_index(drop=True)

    except Exception:
        return None


def safe_calculate_rsi(data, period=14):
    try:
        if len(data) < period + 10:
            return pd.Series([50.0] * len(data), index=data.index)

        close_prices = data['close'].copy()
        delta = close_prices.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        for i in range(period, len(data)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.clip(0, 100).fillna(50)

        return rsi

    except Exception:
        return pd.Series([50.0] * len(data), index=data.index)


def safe_calculate_stoch_rsi(data, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    try:
        min_required = rsi_period + stoch_period + k_period + d_period + 10

        if len(data) < min_required:
            n = len(data)
            return (
                pd.Series([50.0] * n, index=data.index),
                pd.Series([50.0] * n, index=data.index),
                pd.Series([50.0] * n, index=data.index)
            )

        rsi = safe_calculate_rsi(data, rsi_period)
        rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()

        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        stoch_rsi = ((rsi - rsi_min) / rsi_range * 100).fillna(50).clip(0, 100)

        k_line = stoch_rsi.rolling(window=k_period, min_periods=1).mean()
        d_line = k_line.rolling(window=d_period, min_periods=1).mean()

        k_line = k_line.ffill().fillna(50).clip(0, 100)
        d_line = d_line.ffill().fillna(50).clip(0, 100)
        stoch_rsi = stoch_rsi.ffill().fillna(50).clip(0, 100)

        return k_line, d_line, stoch_rsi

    except Exception:
        n = len(data)
        return (
            pd.Series([50.0] * n, index=data.index),
            pd.Series([50.0] * n, index=data.index),
            pd.Series([50.0] * n, index=data.index)
        )


def enhanced_StochRSI_Strategy(data):
    try:
        if len(data) < 100:
            return None

        df = data.copy()

        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                return None

        k_line, d_line, stoch_rsi = safe_calculate_stoch_rsi(df)

        k_below_50 = k_line < 50
        k_above_d = k_line > d_line
        k_was_below_d = k_line.shift(1) <= d_line.shift(1)
        bullish_cross = k_above_d & k_was_below_d
        k_rising = k_line > k_line.shift(1)

        primary_signal = k_below_50 & bullish_cross
        strong_signal = primary_signal & k_rising

        df['k_line'] = k_line
        df['d_line'] = d_line
        df['stoch_rsi'] = stoch_rsi
        df['bullish_cross'] = bullish_cross
        df['k_rising'] = k_rising
        df['primary_signal'] = primary_signal
        df['strong_signal'] = strong_signal
        df['Entry'] = primary_signal

        return df

    except Exception:
        return None


def enhanced_validate_data(data):
    try:
        if data is None or len(data) < 100:
            return False

        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in data.columns:
                return False
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False

        if data[required_columns].isna().any().any():
            return False
        if (data['close'] <= 0).any() or (data['open'] <= 0).any():
            return False
        if (data['high'] < data['low']).any():
            return False

        price_change = data['close'].pct_change().abs()
        if (price_change > 0.5).any():
            return False

        return True

    except Exception:
        return False


def enhanced_get_hist(tv, symbol, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = tv.get_hist(
                symbol=symbol,
                exchange='BIST',
                interval=SELECTED_INTERVAL,
                n_bars=SELECTED_BARS
            )

            if data is None:
                time.sleep(1)
                continue

            cleaned_data = clean_data(data)

            if cleaned_data is None:
                time.sleep(1)
                continue

            if enhanced_validate_data(cleaned_data):
                return cleaned_data

            time.sleep(1)

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def main_enhanced(scan_number=1):
    print(f"\n{'='*60}")
    print(f"🚀 TARAMA #{scan_number} BAŞLIYOR - {get_current_time()}")
    print(f"{'='*60}")

    try:
        tv = TvDatafeed()
        print("✅ TradingView bağlantısı kuruldu")
    except Exception as e:
        print(f"❌ TradingView bağlantı hatası: {e}")
        return

    try:
        hisseler = get_all_symbols(market='turkey')
        hisseler = [s.replace('BIST:', '') for s in hisseler]
        hisseler = [h for h in hisseler if h not in ['REEDR', 'VESTL', 'YAPRK']]
        hisseler = sorted(hisseler)
        print(f"📊 Toplam {len(hisseler)} hisse taranacak")
    except Exception as e:
        print(f"❌ Hisse listesi alınamadı: {e}")
        return

    signals = []
    successful = 0
    failed = 0

    for i, hisse in enumerate(hisseler, 1):
        if i % 50 == 1:
            print(f"📈 [{i}/{len(hisseler)}] İşleniyor...")

        try:
            data = enhanced_get_hist(tv, hisse)

            if data is None:
                failed += 1
                continue

            result = enhanced_StochRSI_Strategy(data)

            if result is None or len(result) < 5:
                failed += 1
                continue

            last = result.iloc[-1]
            prev = result.iloc[-2]

            last_price = float(last['close'])
            k_val = float(last['k_line'])
            d_val = float(last['d_line'])

            # Boolean değerleri güvenli karşılaştır
            current_signal = bool(last['Entry'])
            prev_signal = bool(prev['Entry'])
            new_signal = current_signal and not prev_signal
            strong_signal = bool(last['strong_signal'])

            successful += 1

            if new_signal:
                signal_type = "GÜÇLÜ" if strong_signal else "YENİ"
                signals.append({
                    'hisse': hisse,
                    'fiyat': last_price,
                    'k': k_val,
                    'd': d_val,
                    'tur': signal_type
                })
                print(f"  🚨 SİNYAL: {hisse} - {last_price:.2f}₺ - K:{k_val:.1f} D:{d_val:.1f}")

            time.sleep(0.3)

        except KeyboardInterrupt:
            raise
        except Exception:
            failed += 1

    print(f"\n🎯 SONUÇLAR:")
    print(f"✅ Başarılı: {successful} | ❌ Başarısız: {failed} | 🚨 Sinyal: {len(signals)}")

    msg = (
        f"📊 *BIST StochRSI Tarama Sonuçları*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 *Tarih:* {get_current_time()}\n"
        f"📈 *Interval:* {INTERVAL_NAME}\n"
        f"🔢 *Tarama #:* {scan_number}\n\n"
        f"📋 *Özet:*\n"
        f"   • Toplam Hisse: {len(hisseler)}\n"
        f"   • Başarılı: {successful}\n"
        f"   • Başarısız: {failed}\n"
        f"   • Başarı Oranı: {(successful/len(hisseler)*100):.1f}%\n\n"
    )

    if signals:
        msg += f"🚨 *{len(signals)} SİNYAL TESPİT EDİLDİ!*\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for s in signals:
            emoji = "🔥" if s['tur'] == "GÜÇLÜ" else "📈"
            msg += f"{emoji} *{s['hisse']}* - {s['fiyat']:.2f}₺\n"
            msg += f"   K: {s['k']:.1f} | D: {s['d']:.1f} | Tür: {s['tur']}\n\n"
    else:
        msg += f"ℹ️ *Bu taramada sinyal bulunamadı*\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    msg += f"\n⏰ *Sonraki tarama 30 dakika sonra...*"
    mesaj_at(msg)


def continuous_scan():
    scan_count = 0

    start_msg = (
        f"🤖 *Otomatik Tarama Başladı*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {get_current_time()}\n"
        f"⏰ Saat Dilimi: Türkiye (GMT+3)\n"
        f"📈 Interval: {INTERVAL_NAME}\n"
        f"🔄 Tarama Sıklığı: 30 dakika"
    )
    mesaj_at(start_msg)

    while True:
        scan_count += 1
        main_enhanced(scan_number=scan_count)
        next_time = (datetime.now(TIMEZONE) + pd.Timedelta(minutes=30)).strftime('%H:%M:%S')
        print(f"\n⏳ 30 dakika bekleniyor... Sonraki tarama: {next_time}\n")
        time.sleep(SCAN_INTERVAL_SECONDS)


# ============================
# ANA ÇALIŞTIRMA
# ============================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BIST StochRSI Otomatik Tarayıcı v4.1")
    print(f"📈 Mod: {INTERVAL_NAME} | Otomatik (30dk)")
    print("="*60)

    test_msg = (
        f"✅ *Bot Aktif*\n"
        f"📅 {get_current_time()}\n"
        f"⏰ Türkiye Saati (GMT+3)\n"
        f"📈 {INTERVAL_NAME} otomatik tarama başlıyor..."
    )
    mesaj_at(test_msg, silent=True)

    continuous_scan()
