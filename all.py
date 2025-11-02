import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dateutil.relativedelta import relativedelta


def debug_print(title, data, max_rows=3):
    """Debug information output"""
    print(f"\n=== {title} ===")
    print(f"Total records: {len(data)}")
    if not data.empty:
        print(data.iloc[:max_rows].to_string(index=False))


def parse_date(date_str):
    """Parse YYYY.MM format dates accurately"""
    try:
        str_date = str(date_str).strip()

        if '.' in str_date:
            year_part, month_part = str_date.split('.', 1)
            year = int(year_part)
            month = int(month_part)

            if not (1 <= month <= 12):
                raise ValueError(f"Invalid month: {month}")

            return datetime(year, month, 1)

        return pd.to_datetime(str_date, errors='raise')

    except Exception as e:
        print(f"Date parse failed: {str_date} -> {str(e)}")
        return pd.NaT


def load_data():
    """Load and clean data"""
    try:
        df = pd.read_excel(
            "14-23.xlsx",
            usecols=['date', 'FTPTWK', 'HOURPAY'],
            na_values=["Does not apply", "No Answer", "Not applicable", "N/A", "NA"]
        )
        debug_print("Raw data", df)
    except Exception as e:
        raise ValueError(f"File load failed: {str(e)}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()
    debug_print("Standardized columns", df)

    # Date parsing
    df['DATE'] = df['DATE'].apply(parse_date)
    df = df[df['DATE'].notna()].copy()
    debug_print("After date parsing", df.head(3))

    # Filter employment types
    df['FTPTWK'] = df['FTPTWK'].astype(str).str.strip().str.lower()
    valid_employment = df['FTPTWK'].isin(['full-time', 'part-time'])
    df = df[valid_employment].copy()
    debug_print("After employment filter", df.head(3))

    # Process hourly pay
    df['HOURPAY'] = pd.to_numeric(df['HOURPAY'], errors='coerce')
    valid_pay = (df['HOURPAY'] > 0.5) & (df['HOURPAY'] < 1000)
    df = df[valid_pay].copy()
    debug_print("After pay filtering", df.head(3))

    if df.empty:
        print("\nWarning: No valid data remaining!")
        return df

    # Dataset statistics
    print("\nFinal dataset stats:")
    print(f"Date range: {df['DATE'].min().strftime('%Y-%m')} to {df['DATE'].max().strftime('%Y-%m')}")
    print("Employment type distribution:")
    print(df['FTPTWK'].value_counts().to_string())
    print(f"Hourly pay range: £{df['HOURPAY'].min():.2f} - £{df['HOURPAY'].max():.2f}")

    return df


def generate_monthly_ranges(df):
    """Generate complete monthly range"""
    start_date = df['DATE'].min().to_pydatetime().replace(day=1)
    end_date = df['DATE'].max().to_pydatetime().replace(day=1)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += relativedelta(months=1)
    return dates


def plot_combined_histogram(ax, data, title):
    """Plot combined histogram with different colors"""
    if data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12, fontname='Times New Roman')
        return

    try:
        # Split data
        ft = data[data['FTPTWK'] == 'full-time']['HOURPAY']
        pt = data[data['FTPTWK'] == 'part-time']['HOURPAY']

        # Create common bins
        bins = np.linspace(0, 50, 21)

        # Plot both histograms
        ax.hist([ft, pt], bins=bins,
                color=['#1f77b4', '#ff7f0e'],
                alpha=0.6,  # 调整透明度
                label=['Full-time', 'Part-time'])

        # Add legend and labels
        ax.legend(loc='upper right', fontsize=10, prop={'family': 'Times New Roman'})
        ax.set_xlabel('Hourly Pay (£)', fontsize=12, fontname='Times New Roman')
        ax.set_ylabel('Count', fontsize=12, fontname='Times New Roman')
        ax.grid(True, alpha=0.2)  # 调整网格线透明度
        ax.set_title(title, fontsize=12, pad=15, fontname='Times New Roman', fontweight='bold')

        # 设置刻度字体
        for tick in ax.get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(10)

    except Exception as e:
        ax.text(0.5, 0.5, f'Plot Error\n{str(e)}', color='red', fontsize=12, fontname='Times New Roman')


def generate_pdf_report(df):
    """Generate PDF report"""
    all_months = generate_monthly_ranges(df)

    with PdfPages('Combined_Hourly_Pay_Report.pdf') as pdf:
        for year in sorted({d.year for d in all_months}):
            fig, axs = plt.subplots(3, 4, figsize=(21, 29.7))  # A4 size
            plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.93, bottom=0.05)  # 增加子图间距

            for month_idx in range(12):
                row = month_idx // 4
                col = month_idx % 4
                ax = axs[row, col]

                target_month = datetime(year, month_idx + 1, 1)
                month_data = df[
                    (df['DATE'].dt.year == year) &
                    (df['DATE'].dt.month == month_idx + 1)
                    ]

                title = target_month.strftime("%Y-%m")
                ft_count = len(month_data[month_data['FTPTWK'] == 'full-time'])
                pt_count = len(month_data[month_data['FTPTWK'] == 'part-time'])
                title += f"\nFT:{ft_count} PT:{pt_count}"

                plot_combined_histogram(ax, month_data, title)

            fig.suptitle(f"Combined Hourly Pay Distribution - {year}",
                         y=0.97, fontsize=16,
                         fontweight='bold', fontname='Times New Roman')

            plt.figtext(0.5, 0.02,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: 14-23.xlsx",
                        ha='center', fontsize=10, fontname='Times New Roman')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def generate_pdf_report(df):
    """Generate PDF report"""
    all_months = generate_monthly_ranges(df)

    with PdfPages('Combined_Hourly_Pay_Report.pdf') as pdf:
        for year in sorted({d.year for d in all_months}):
            fig, axs = plt.subplots(3, 4, figsize=(21, 29.7))  # A4 size
            plt.subplots_adjust(wspace=0.25, hspace=0.35, top=0.93, bottom=0.05)

            for month_idx in range(12):
                row = month_idx // 4
                col = month_idx % 4
                ax = axs[row, col]

                target_month = datetime(year, month_idx + 1, 1)
                month_data = df[
                    (df['DATE'].dt.year == year) &
                    (df['DATE'].dt.month == month_idx + 1)
                    ]

                title = target_month.strftime("%Y-%m")
                ft_count = len(month_data[month_data['FTPTWK'] == 'full-time'])
                pt_count = len(month_data[month_data['FTPTWK'] == 'part-time'])
                title += f"\nFT:{ft_count} PT:{pt_count}"

                plot_combined_histogram(ax, month_data, title)

            fig.suptitle(f"Combined Hourly Pay Distribution - {year}",
                         y=0.97, fontsize=14,
                         fontweight='bold')

            plt.figtext(0.5, 0.02,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: 14-23.xlsx",
                        ha='center', fontsize=8)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def main():
    try:
        print("=== Processing Data ===")
        df = load_data()

        if not df.empty:
            print("\n=== Generating Report ===")
            generate_pdf_report(df)
            print("=== Report Complete ===")
            print("Output File: Combined_Hourly_Pay_Report.pdf")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Troubleshooting:")
        print("1. Verify Excel file path")
        print("2. Check DATE column format (should be YYYY.MM)")
        print("3. Validate HOURPAY column contains numbers")


if __name__ == "__main__":
    main()