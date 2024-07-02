from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from pathlib import Path
import csp_2 as cl
import datetime as dt
import numpy as np

toc = Path("assets/brand_assets/toc.png")
toc_ex = Path("assets/brand_assets/toc_ex.png")
blank = Path("assets/brand_assets/blank.png")
firm_div = Path("assets/brand_assets/firm_slide.png")
about = Path("assets/brand_assets/about.png")
csp_div = Path("assets/brand_assets/csp_div.png")
csp_over = Path("assets/brand_assets/csp_over.png")
csp_over_tab = Path("assets/brand_assets/csp_over_tab.png")
image_path = Path("assets/brand_assets/Lido_Logo_RGB-Full-Color.png")
cc_div = Path("assets/brand_assets/cc_div.png")
hedge_div = Path("assets/brand_assets/hedge_div.png")
sef_div = Path("assets/brand_assets/sef_div.png")
desc_path = Path("assets/brand_assets/disc.png")
hedge_over = Path("assets/brand_assets/hedge_over.png")
cc_over = Path("assets/brand_assets/cc_over.png")
sef_over = Path("assets/brand_assets/sef_over.png")
sef_proc = Path("assets/brand_assets/sef_proc.png")
disc_div = Path("assets/brand_assets/disc_div.png")
notes = Path("assets/brand_assets/notes.png")
defin = Path("assets/brand_assets/def.png")
discs = Path("assets/brand_assets/disc.png")


try:
    pdfmetrics.registerFont(TTFont('Noe', '/Users/jgarden/NoeDisplay-Bold.ttf'))  # change to jeffreygarden for mac
except:
    pdfmetrics.registerFont(TTFont('Noe', '/Users/jeffreygarden/NoeDisplay-Bold.ttf'))

try:
    pdfmetrics.registerFont(
        TTFont('sans_l', '/Users/jgarden/UntitledSans-Light.ttf'))  # change to jeffreygarden for mac
except:
    pdfmetrics.registerFont(TTFont('sans_l', '/Users/jeffreygarden/UntitledSans-Light.ttf'))  #

try:
    pdfmetrics.registerFont(
        TTFont('sans_b', '/Users/jgarden/UntitledSans-Bold.ttf'))  # change to jeffreygarden for mac
except:
    pdfmetrics.registerFont(TTFont('sans_b', '/Users/jeffreygarden/UntitledSans-Bold.ttf'))  #

# Define global styles
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'Title',
    parent=styles['Title'],
    fontName='Noe',
    fontSize=28,
    leading=34,
    textColor=colors.HexColor('#000000'),
    alignment=0  # Left align
)

subtitle_style = ParagraphStyle(
    'Subtitle',
    parent=styles['Normal'],
    fontName='sans_l',
    fontSize=12,
    leading=13,
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='sans_l',
    fontSize=15,
    leading=17,
    fontWeight='bold'
)

table_style = ParagraphStyle(
    'Table',
    parent=styles['Normal'],
    fontName='sans_l',
    fontSize=14,
    leading=16,
)

footer_style = ParagraphStyle(
    'Footer',
    parent=styles['Normal'],
    fontName='sans_l',
    fontSize=9,
    leading=11,
    alignment=0  # Center align
)

emph = ParagraphStyle(
    'emph',
    parent=styles['Normal'],
    fontName='sans_b',
    fontSize=14,
    leading=16,
)

class CspDeck:

    def __init__(self, csp_client:cl.CspClient):
        self.csp_client = csp_client


    def _generate_csp_report(self):
        client_name = self.csp_client.client_name
        pdf_path = f"decks/{client_name}_csp_report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=(13.333 * inch, 7.5 * inch))
        return c

    def _gen_content_page(self, c: canvas.Canvas):
        c.showPage()
        # Set background color
        c.drawImage(blank, 0, 0, width=13.333 * inch, height= 7.5 * inch,
                    mask='auto')  # 'mask="auto"' for transparency
        return c

    def _add_title_page(self, c: canvas.Canvas):
        # Add background image
        path = Path("assets/brand_assets/title_page.png")
        c.drawImage(path, 0, 0, width=13.333 * inch, height=7.5 * inch, mask='auto')  # 'mask="auto"' for transparency

        # Add prepared text
        prepared_text = (f"Prepared for: {self.csp_client.client_name}<br/>"
                         f"Stock: {self.csp_client.sym.upper()}<br/>"
                         f"Report Date: {dt.datetime.today().strftime('%Y-%m-%d')}<br/>")
        prepared_paragraph = Paragraph(prepared_text, ParagraphStyle(name='prepared', fontName='sans_l', fontSize=12, leading=14, textColor=colors.HexColor('#FFFFFF')))

        # Draw prepared text at the bottom right corner
        text_x = 13.333 * inch - 2.5 * inch  # Adjust as needed
        text_y = 0.5 * inch  # Adjust as needed
        prepared_paragraph.wrapOn(c, 2.5 * inch, 1 * inch)
        prepared_paragraph.drawOn(c, text_x, text_y)

        return c


    def _create_summary_analysis_page(self, c):
        # Generate content page with the image
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()
        value = np.round((self.csp_client.shares * self.csp_client.last)/1000000, 2)
        shares = self.csp_client.shares
        date = dt.datetime.today().strftime('%Y-%m-%d')
        days = int(self.csp_client.cov_call.adj_time_left)
        # Title and subtitle
        title = f"Summary Analysis – {ticker} (Hypothetical)"
        subtitle = (f"For an investor with a ~${value:,.2f}mm ({shares} share) position in {ticker}<br/>"
                    f"Based on current market conditions and options pricing (as of {date}), an investor implementing Lido’s<br/>"
                    "Concentrated Stock Program could expect the following outcomes to meet various client needs:")

        title_paragraph = Paragraph(title, title_style)
        subtitle_paragraph = Paragraph(subtitle, subtitle_style)

        # Calculate table width and margins
        total_table_width = 12.67 * inch  # 13.333 inches - 2/3 inch margin (1/3 inch on each side)
        col_widths = [total_table_width / 3] * 3  # Equal column widths

        # Table data with Paragraphs for multi-line text
        max_gain = self.csp_client.gap.upside
        trade_gap = self.csp_client.gap.ds_before * -1
        protection = self.csp_client.gap.protection * -1
        prem = self.csp_client.cov_call.premium
        prem_percent = self.csp_client.cov_call.premium_percent
        otm = (self.csp_client.cov_call.S_CALL['STRIKE_PRC']/self.csp_client.last) - 1
        sef_short_call = self.csp_client.sef.collar[self.csp_client.sef.collar['Trade'] == 'SCO'].iloc[0]['STRIKE_PRC']
        sef_upside = (sef_short_call/self.csp_client.last)-1
        sef_put = self.csp_client.sef.collar[self.csp_client.sef.collar['Trade'] == 'BPO'].iloc[0]['STRIKE_PRC']
        sef_downside = ((self.csp_client.last/sef_put)-1)
        sp_long_call = self.csp_client.sef.synthetic[self.csp_client.sef.synthetic['Trade']=='BCO'].iloc[0][
            'STRIKE_PRC']
        sp_short_put = self.csp_client.sef.synthetic[self.csp_client.sef.synthetic['Trade'] == 'SPO'].iloc[0][
            'STRIKE_PRC']
        synth_up = (sp_long_call  / self.csp_client.sef.last_mkt)-1
        synth_risk = (sp_short_put / self.csp_client.sef.last_mkt)-1

        data = [
            [Paragraph("Goal", table_header_style), Paragraph("Strategy", table_header_style),
             Paragraph("Defined Outcome", table_header_style)],
            [Paragraph("Hedge Risk", table_style), Paragraph("Cap and Cushion Strategy", table_style),
             Paragraph(f"Max. Gain: {max_gain:,.1%}<br/>Downside Gap: {trade_gap:,.1%}<br/>Protection after Gap: {protection:,.1%}<br/>Term: 1 year",
                       table_style)],
            [Paragraph("Generate Cash Flow for Strategic Exit", table_style),
             Paragraph("Covered Call Strategy", table_style),
             Paragraph(
                 f"Initial Premium Received: ${prem:,.2f} ({prem_percent:,.1%})<br/>Call OTM Percent: {otm:,.1%}<br/>Annualized Premium Potential ($): ${prem*365/days/1000:,.2f}k<br/>Annualized Premium Potential (%): {prem_percent*4:,.1%}<br/>Term: Ongoing",
                 table_style)],
            [Paragraph("Transfer Risk", table_style), Paragraph("Synthetic Exchange Fund Strategy", table_style),
             Paragraph(
                 f"Max. gain on {ticker}: {sef_upside:,.1%}<br/>Max. loss on {ticker}: {sef_downside:,.1%}<br/>Gains on S&amp;P 500 starts at: {synth_up:,.1%}<br/>Risk on S&amp;P 500 buffered through: {synth_risk:,.1%}<br/>Term: 6 months",
                 table_style)]
        ]

        # Create the table
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A7B43F')),  # Header fill color
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), 'sans_l'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        # Footer text
        footer_text = ("Goals and strategies listed are not an exhaustive list of Lido’s capabilities. "
                       "All option transactions produce tax consequences. Lido does not provide tax advice.<br/>Note: The figures listed above are hypothetical in nature and based on current market conditions as of 06/23/2024. "
                       "Projections and estimates of outcomes after engaging Lido will be different.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        table_y = 2 * inch
        footer_y = (7.5 * inch) - (6.5 * inch)

        # Draw elements
        title_paragraph.wrapOn(c, total_table_width, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, total_table_width, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        table.wrapOn(c, total_table_width, inch)
        table.drawOn(c, 0.333 * inch, table_y)

        footer_paragraph.wrapOn(c, total_table_width, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)
        return c


    def create_cc_page(self, c):
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()
        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle and content
        subtitle = "Considerations – Cap & Cushion"
        subtitle_paragraph = Paragraph(subtitle,
                                       style=ParagraphStyle(name='cons', fontName='sans_b', fontSize=18, leading=16, ))

        emph = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName='sans_b',
            fontSize=14,
            leading=16,
            alignment=0  # Center align
        )

        desired_outcome = ("<font color='#818735'><b>Desired Outcome:</b></font><br/>"
                           "Reduce single stock risk, define investment outcome, maintain current position")
        target_return = ("<font color='#818735'><b>Target Return (absolute or annualized):</b></font><br/>"
                         "Price return after which investor is willing to forego continued upside appreciation in the underlying stock")
        target_protection = ("<font color='#818735'><b>Target Protection (absolute or annualized):</b></font><br/>"
                             "A desired buffer against losses may be a factor in strike price selection")
        expiration_date = ("<font color='#818735'><b>Expiration date:</b></font><br/>"
                           "Based on available options expirations, an investor may choose to define their outcome for a specific period")

        desired_outcome_paragraph = Paragraph(desired_outcome, emph)
        target_return_paragraph = Paragraph(target_return, table_style)
        target_protection_paragraph = Paragraph(target_protection, table_style)
        expiration_date_paragraph = Paragraph(expiration_date, table_style)

        # Add image
        image_width = 7.5 * inch
        image_height = 4.29 * inch
        image_x = 0.333 * inch
        image_y = 2 * inch

        date = dt.datetime.today().strftime('%m/%d/%Y')

        # Footer text
        footer_text = f"Data Source: Thomson Reuters, as of market close {date}. Visualization created by Lido."
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        desired_outcome_y = 7.5 * inch - 2.4 * inch
        target_return_y = 7.5 * inch - 3.4 * inch
        target_protection_y = 7.5 * inch - 4.4 * inch
        expiration_date_y = 7.5 * inch - 5.4 * inch
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 6 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 6 * inch, inch)
        subtitle_paragraph.drawOn(c, 8.6 * inch, subtitle_y)

        cc_price_path = Path(f"assets/images/{ticker}_2yr_price_chart_CapCush.png")

        c.drawImage(cc_price_path, image_x, image_y, width=image_width, height=image_height)

        desired_outcome_paragraph.wrapOn(c, 4 * inch, inch)
        desired_outcome_paragraph.drawOn(c, 8.6 * inch, desired_outcome_y)

        target_return_paragraph.wrapOn(c, 4 * inch, inch)
        target_return_paragraph.drawOn(c, 8.6 * inch, target_return_y)

        target_protection_paragraph.wrapOn(c, 4 * inch, inch)
        target_protection_paragraph.drawOn(c, 8.6 * inch, target_protection_y)

        expiration_date_paragraph.wrapOn(c, 4 * inch, inch)
        expiration_date_paragraph.drawOn(c, 8.6 * inch, expiration_date_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_payoff_diagram_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)

        ticker = self.csp_client.sym.upper()
        max_gain = self.csp_client.gap.upside
        trade_gap = self.csp_client.gap.ds_before * -1
        protection = self.csp_client.gap.protection * -1
        delta = self.csp_client.gap.trade_delta
        time = self.csp_client.gap.adj_time_left
        cost = self.csp_client.gap.net_opt_cost_dlrs * self.csp_client.gap.contracts
        stk_val = self.csp_client.gap.port_val
        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle
        subtitle = "<b>Payoff diagram at options expiration</b>"
        subtitle_paragraph = Paragraph(subtitle, emph)

        right_style = ParagraphStyle(
            'right',
            parent=styles['Normal'],
            fontSize=16,
            leading=24,
        )


        # Right-side content
        right_content = (f"<b>Stock Value:</b> ${stk_val:,.2f}<br/>"
                         f"<b>Net Options Cost:</b> ${cost:,.2f}<br/>"
                         f"<b>Max Upside:</b> {max_gain:,.1%}<br/>"
                         f"<b>Gap Before Protection:</b> {trade_gap:,.1%}<br/>"
                         f"<b>Protection:</b> {protection*1:,.1%}<br/>"
                         f"<b>Trade Length:</b> {time} Days<br/>"
                         f"<b>Risk Reduction:</b> {delta:,.2%}<br/>")
        right_content_paragraph = Paragraph(right_content, right_style)

        date = dt.datetime.today().strftime('%m/%d/%Y')
        # Footer text
        footer_text = (
            "Note: Performance illustrated on this page is hypothetical in nature based on the price of the underlying position and market conditions at the time the report was generated. "
            f"Strategy returns are illustrated gross of Lido’s management fees. Data source: Thomson Reuters, as of market close {date}.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Add images
        image_width = 7.5 * inch
        image_height = 3.5 * inch
        image_x = 0.333 * inch
        image_y = 2.5 * inch

        # Second image position
        table_image_width = 12 * inch
        table_image_height = 1.5 * inch
        table_image_x = 0.333 * inch
        table_image_y = 1 * inch

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        right_content_y = 7.5 * inch - 3 * inch
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 6 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 6 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)
        cc_payoff = Path(f"assets/images/{ticker}_gap_trade_payoff_plot.png")
        cc_table = Path(f"assets/images/return_at_expiration_table_{ticker}_gap.png")

        c.drawImage(cc_payoff, image_x, image_y, width=image_width, height=image_height)
        c.drawImage(cc_table, table_image_x, table_image_y, width=table_image_width, height=table_image_height)

        right_content_paragraph.wrapOn(c, 4 * inch, inch)
        right_content_paragraph.drawOn(c, 8.6 * inch, right_content_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_covered_calls_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()

        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle and content
        subtitle = "Considerations – Covered Calls"
        subtitle_paragraph = Paragraph(subtitle,
                                       style=ParagraphStyle(name='cons', fontName='sans_b', fontSize=18, leading=16))

        # Styles for emphasized text
        emph = ParagraphStyle(
            'Emph',
            parent=styles['Normal'],
            fontName='sans_b',
            fontSize=14,
            leading=16,
            alignment=0
        )

        # Content with specific color
        desired_outcome = ("<font color='#A7B43F'><b>Desired Outcome:</b></font><br/>"
                           "Overall Risk Reduction plus Cash Flow Enhancement")
        target_price = ("<font color='#A7B43F'><b>Target Price:</b></font><br/>"
                        "Price at which investor is willing to forego continued upside appreciation in the underlying stock")
        target_income = ("<font color='#A7B43F'><b>Target Income:</b></font><br/>"
                         "Desired annualized cash flow may be a factor in strike price selection")

        desired_outcome_paragraph = Paragraph(desired_outcome, emph)
        target_price_paragraph = Paragraph(target_price, table_style)
        target_income_paragraph = Paragraph(target_income, table_style)

        # Add image
        image_path = Path(f"assets/images/{ticker}_2yr_price_chart_CovdCall.png")
        image_width = 7.5 * inch
        image_height = 4.29 * inch
        image_x = 0.333 * inch
        image_y = 2 * inch

        date = dt.datetime.today().strftime('%m/%d/%Y')
        # Footer text
        footer_text = f"Data Source: Thomson Reuters, as of market close {date}. Visualization created by Lido."
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        desired_outcome_y = 7.5 * inch - 2.4 * inch
        target_price_y = 7.5 * inch - 3.4 * inch
        target_income_y = 7.5 * inch - 4.4 * inch
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 6 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 6 * inch, inch)
        subtitle_paragraph.drawOn(c, 8.6 * inch, subtitle_y)

        c.drawImage(image_path, image_x, image_y, width=image_width, height=image_height)

        desired_outcome_paragraph.wrapOn(c, 4 * inch, inch)
        desired_outcome_paragraph.drawOn(c, 8.6 * inch, desired_outcome_y)

        target_price_paragraph.wrapOn(c, 4 * inch, inch)
        target_price_paragraph.drawOn(c, 8.6 * inch, target_price_y)

        target_income_paragraph.wrapOn(c, 4 * inch, inch)
        target_income_paragraph.drawOn(c, 8.6 * inch, target_income_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_covcall_payoff_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)

        ticker = self.csp_client.sym.upper()
        prem = self.csp_client.cov_call.premium
        prem_percent = self.csp_client.cov_call.premium_percent
        otm = (self.csp_client.cov_call.S_CALL['STRIKE_PRC'] / self.csp_client.last) - 1
        share_val = self.csp_client.cov_call.share_value
        days = int(self.csp_client.cov_call.adj_time_left)
        delta = self.csp_client.cov_call.S_CALL['DELTA']
        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle
        subtitle = "<b>Payoff diagram at options expiration</b> Overwrite Percent: 100%"
        subtitle_paragraph = Paragraph(subtitle, emph)

        right_style = ParagraphStyle(
            'right',
            parent=styles['Normal'],
            fontSize=16,
            leading=24,
        )

        # Right-side content
        right_content = (f"<b>Stock Value:</b> ${share_val:,.2f}<br/>"
                         f"<b>Premium Collected:</b> ${prem:,.2f}<br/>"
                         f"<b>Distance to Strike:</b> {otm:,.1%}<br/>"
                         f"<b>Trade Length:</b> {days} Days<br/>"
                         f"<b>Annualized Premium:</b> ${prem*365/days:,.2f}<br/>"
                         f"<b>Risk Reduction:</b> {delta:,.2%}<br/>")
        right_content_paragraph = Paragraph(right_content, right_style)

        date = dt.datetime.today().strftime('%m/%d/%Y')
        # Footer text
        footer_text = (
            "Note: Performance illustrated on this page is hypothetical in nature based on the price of the underlying position and market conditions at the time the report was generated. "
            f"Strategy returns are illustrated gross of Lido’s management fees. Data source: Thomson Reuters, as of market close {date}.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Add images
        new_image_path = Path(f"assets/images/{ticker}_covd_call_payoff.png")  # Update with the actual path
        image_width = 7.5 * inch
        image_height = 3.5 * inch
        image_x = 0.333 * inch
        image_y = 2.5 * inch

        # Second image position
        new_table_image_path = Path(f"assets/images/return_at_expiration_table_cov_{ticker}.png")  # Update with the actual path
        table_image_width = 12 * inch
        table_image_height = 1.5 * inch
        table_image_x = 0.333 * inch
        table_image_y = 1 * inch

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        right_content_y = 7.5 * inch - 3 * inch
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 6 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 6 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        c.drawImage(new_image_path, image_x, image_y, width=image_width, height=image_height)
        c.drawImage(new_table_image_path, table_image_x, table_image_y, width=table_image_width, height=table_image_height)

        right_content_paragraph.wrapOn(c, 4 * inch, inch)
        right_content_paragraph.drawOn(c, 8.6 * inch, right_content_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_tax_neutral_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()
        cb = self.csp_client.cost_basis_per_share

        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle
        subtitle = f"<b>Amount of Liquidation - {ticker} Tax Neutral Share Sale</b>"
        subtitle_paragraph = Paragraph(subtitle, emph)

        right_style = ParagraphStyle(
            'right',
            parent=styles['Normal'],
            fontSize=14,
            leading=16,
        )

        # Right-side content
        right_content = (
            "The Covered Call strategy can be used to strategically exit a concentrated stock position over time, freeing up capital to be redeployed into more diversified investments.<br/><br/>"
            "“Tax Neutral” refers to a share sale with no “out-of-pocket” expense.<br/><br/>"
            "<b>• If the options make money</b>, shares can be sold such that the long-term capital gains taxes due are equal to the after-tax dollar amount of the options profit.<br/><br/>"
            "<b>• If the options lose money</b>, the short-term capital loss can be used to offset the long-term capital gain of a share sale of a specific amount.")
        right_content_paragraph = Paragraph(right_content, right_style)

        # Footer text
        footer_text = (
            "Lido does not provide tax advice. Each option transaction has tax consequences. Consult your tax professional.<br/>"
            f"* The hypothetical example shown here assumes a cost-basis on {ticker} of {cb} per share, a long-term capital gains tax rate of 23.8% and a short-term capital gains tax rate of 37%.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Add images
        new_image_path = Path(f"assets/images/{ticker}_strategic_exit.png")  # Update with the actual path
        image_width = 7.5 * inch
        image_height = 4.29 * inch
        image_x = 0.333 * inch
        image_y = 2 * inch

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.4 * inch)
        right_content_y = 7.5 * inch - 5 * inch
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 6 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 6 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        c.drawImage(new_image_path, image_x, image_y, width=image_width, height=image_height)

        right_content_paragraph.wrapOn(c, 3.75 * inch, inch)
        right_content_paragraph.drawOn(c, 8.6 * inch, right_content_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_sef_strategy_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()

        # Title
        title = "NVDA Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle
        subtitle = (
            f"By utilizing the options markets, we create two exposures with defined outcomes – one for the existing position on {ticker} (collar trade) and one for a new synthetic position in SPY (long Calls, short Puts).")
        subtitle_paragraph = Paragraph(subtitle, ParagraphStyle(name='Subtitle', fontName='sans_l', fontSize=14, leading=18,
                                                                textColor=colors.HexColor('#4569A8')))

        # Left-side content
        left_content = (f"1. Define {ticker} Risk<br/>"
                        f"• Buy Put options on {ticker}<br/>"
                        f"• Sell Call options on {ticker}<br/>"
                        f"• Combined position is a collar strategy to define risk on {ticker} shares")
        left_content_paragraph = Paragraph(left_content, table_style)

        # Right-side content
        right_content = ("2. Shift Toward Broad Market Exposure<br/>"
                         "• Sell Put options on the SPY ETF<br/>"
                         "• Buy Call options on the SPY ETF<br/>"
                         "• Provides synthetic exposure the  S&amp;P 500")
        right_content_paragraph = Paragraph(right_content, table_style)

        # Additional text
        additional_text = (
            f"• In the combined portfolio, the concentrated position in {ticker} is fully hedged<br/>"
            f"• Dividends on {ticker} are retained and may be pre-spent to maximize the upside potential of the collar trade<br/>"
            "• Downside risk transfer to SPY below the SPY short Put strike; unlimited broad market upside potential starts at the SPY long Call strike")
        additional_text_paragraph = Paragraph(additional_text, table_style)

        date = dt.date.today().strftime("%m/%d/%Y")
        # Footer text
        footer_text = (
            "Note: Performance illustrated on this page is hypothetical in nature based on the price of the underlying position and market conditions at the time the report was generated. "
            f"Strategy returns are illustrated gross of Lido’s management fees. Data source: Thomson Reuters, as of market close {date}.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Add images
        nvda_image_path = Path(f"assets/images/{ticker}_collar_payoff.png")  # Update with the actual path
        spy_image_path = Path("assets/images/SPY_synthetic_payoff_NVDA.png")  # Update with the actual path
        nvda_image_width = 5.25 * inch
        nvda_image_height = 2.25 * inch
        nvda_image_x = 0.333 * inch
        nvda_image_y = 2.5 * inch
        spy_image_width = 5.25 * inch
        spy_image_height = 2.25 * inch
        spy_image_x = 6.5 * inch
        spy_image_y = 2.5 * inch

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.2 * inch)
        right_y = (7.5 * inch) - (2.2 * inch)
        left_y = (7.5 * inch) - (2.4 * inch)
        footer_y = 7.5 * inch - 6.5 * inch
        additional_text_y = 7.5 * inch - 5.75 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 12 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 12 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        left_content_paragraph.wrapOn(c, 6 * inch, inch)
        left_content_paragraph.drawOn(c, 0.333 * inch, left_y)

        right_content_paragraph.wrapOn(c, 6 * inch, inch)
        right_content_paragraph.drawOn(c, 6.8333 * inch, right_y)

        c.drawImage(nvda_image_path, nvda_image_x, nvda_image_y, width=nvda_image_width, height=nvda_image_height)
        c.drawImage(spy_image_path, spy_image_x, spy_image_y, width=spy_image_width, height=spy_image_height)

        additional_text_paragraph.wrapOn(c, 12 * inch, inch)
        additional_text_paragraph.drawOn(c, 0.333 * inch, additional_text_y)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c


    def create_stock_sef_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)
        ticker = self.csp_client.sym.upper()
        sef_short_call = self.csp_client.sef.collar[self.csp_client.sef.collar['Trade'] == 'SCO'].iloc[0]
        # sef_upside = (sef_short_call/self.csp_client.last)-1
        sef_put = self.csp_client.sef.collar[self.csp_client.sef.collar['Trade'] == 'BPO'].iloc[0]
        # sef_downside = ((self.csp_client.last/sef_put)-1)
        sp_long_call = self.csp_client.sef.synthetic[self.csp_client.sef.synthetic['Trade']=='BCO'].iloc[0]
        sp_short_put = self.csp_client.sef.synthetic[self.csp_client.sef.synthetic['Trade'] == 'SPO'].iloc[0]
        # synth_up = (sp_long_call  / self.csp_client.sef.last_mkt)-1
        # synth_risk = (sp_short_put / self.csp_client.sef.last_mkt)-1
        port_val = self.csp_client.sef.port_val
        shares =self.csp_client.sef.shares
        col_cost = self.csp_client.sef.collar_cost
        synt_cost = self.csp_client.sef.synthetic_cost

        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        date = dt.date.today().strftime("%m/%d/%Y")

        # Subtitle
        subtitle = (
            f"Based on current market conditions (as of {date}), a collar trade on a ~${port_val/1000000:,.2f}mm ({shares} share) position in {ticker} could be implemented in the following manner.")
        subtitle_paragraph = Paragraph(subtitle, ParagraphStyle(name='Subtitle', fontName='sans_l', fontSize=14, leading=18,
                                                                textColor=colors.HexColor('#4569A8')))

        # Left-side content
        left_content = (f"1. Define {ticker} Risk<br/>"
                        f"• Buy Put options on {ticker}<br/>"
                        f"• Sell Call options on {ticker}<br/>"
                        f"• Net cost of {ticker} options: ${col_cost:,.2f}<br/>")
        left_content_paragraph = Paragraph(left_content, table_style)

        # Right-side content
        right_content = ("2. Shift Toward Broad Market Exposure<br/>"
                         "• Sell Put options on the SPY ETF<br/>"
                         "• Buy Call options on the SPY ETF<br/>"
                         f"• Net cost of SPY options: ${synt_cost:,.2f}<br/>")
        right_content_paragraph = Paragraph(right_content, table_style)

        date = dt.date.today().strftime("%m/%d/%Y")

        # Footer text
        footer_text = (
            "Note: Performance illustrated on this page is hypothetical in nature based on the price of the underlying position and market conditions at the time the report was generated. "
            f"Strategy returns are illustrated gross of Lido’s management fees. Data source: Thomson Reuters, as of market close {date}.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Tables data
        nvda_table_data = [
            ["Underlier", ticker],
            ["Last", f"{self.csp_client.last}"],
            ["Div. Yield", f"{self.csp_client.dividend}"],
            ["Shares Included", shares],
            ["Value", f"${port_val:,.2f}"],
        ]

        spy_table_data = [
            ["Synthetic", "SPY"],
            ["SPY Last", f"{self.csp_client.sef.last_mkt}"],
            ["SPY Equivalent Shrs", f"~{int(port_val//self.csp_client.sef.last_mkt)}"],
            ["SPY Value", f"${int(port_val//self.csp_client.sef.last_mkt) * self.csp_client.sef.last_mkt:,.2f}"],
            ["Spy Equiv Contracts", f"{self.csp_client.sef.spy_contracts}"],
        ]

        col_contracts = (port_val/self.csp_client.last)//100

        nvda_option_table_data = [
            ["Expiry", "Option", "Strike", "Contracts", "Premium", "Cost($)"],
            [f"{sef_put['EXPIR_DATE']}", "PUT", f"{sef_put['STRIKE_PRC']}", f"{col_contracts}", f"${sef_put['MID']:,.2f}", f"${sef_put['MID'] * col_contracts * 100:,.2f}"],
            [f"{sef_short_call['EXPIR_DATE']}", "CALL", f"{sef_short_call['STRIKE_PRC']}", f"{col_contracts}", f"${sef_short_call['MID']:,.2f}", f"{sef_short_call['MID']* col_contracts * 100:,.2f}"]
        ]

        spy_option_table_data = [
            ["Expiry", "Option", "Strike", "Contracts", "Premium", "Cost($)"],
            [f"{sp_short_put['EXPIR_DATE']}", "PUT", f"{sp_short_put['STRIKE_PRC']}", f"{self.csp_client.sef.spy_contracts}", f"${sp_short_put['MID']:,.2f}", f"${sp_short_put['MID'] * self.csp_client.sef.spy_contracts * 100:,.2f}"],
            [f"{sp_short_put['EXPIR_DATE']}", "CALL", f"{sp_long_call['STRIKE_PRC']}", f"{self.csp_client.sef.spy_contracts}", f"${sp_long_call['MID']:,.2f}", f"${sp_long_call['MID'] * self.csp_client.sef.spy_contracts * 100:,.2f}"]
        ]

        # Create the tables
        nvda_table = Table(nvda_table_data, colWidths=[1.5 * inch, 2 * inch])
        spy_table = Table(spy_table_data, colWidths=[1.5 * inch, 2 * inch])
        nvda_option_table = Table(nvda_option_table_data, colWidths=[1 * inch] * 6)
        spy_option_table = Table(spy_option_table_data, colWidths=[1 * inch] * 6)

        table_style_grid_1 = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3DA9F')),  # Header fill color
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'sans_l'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D3DA9F')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
        ])

        table_style_grid_2 = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#639DD9')),  # Header fill color
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'sans_l'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
        ])

        nvda_table.setStyle(table_style_grid_1)
        spy_table.setStyle(table_style_grid_1)
        nvda_option_table.setStyle(table_style_grid_2)
        spy_option_table.setStyle(table_style_grid_2)

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.2 * inch)
        left_content_y = (7.5 * inch) - (2.2 * inch)
        right_content_y = (7.5 * inch) - (2.2 * inch)
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 12 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 12 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        left_content_paragraph.wrapOn(c, 6 * inch, inch)
        left_content_paragraph.drawOn(c, 0.333 * inch, left_content_y)

        right_content_paragraph.wrapOn(c, 6 * inch, inch)
        right_content_paragraph.drawOn(c, 6.8333 * inch, right_content_y)

        # Draw tables
        nvda_table.wrapOn(c, 2.5 * inch, 2.5 * inch)
        nvda_table.drawOn(c, 0.333 * inch, left_content_y - 1.5 * inch)

        spy_table.wrapOn(c, 2.5 * inch, 2.5 * inch)
        spy_table.drawOn(c, 6.8333 * inch, right_content_y - 1.5 * inch)

        nvda_option_table.wrapOn(c, 6 * inch, 1.5 * inch)
        nvda_option_table.drawOn(c, 0.333 * inch, left_content_y - 3 * inch)

        spy_option_table.wrapOn(c, 6 * inch, 1.5 * inch)
        spy_option_table.drawOn(c, 6.83 * inch, left_content_y - 3 * inch)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c

    def create_sef_matrix_page(self, c):
        # Set the background color and generate content page
        c = self._gen_content_page(c)

        ticker = self.csp_client.sym.upper()
        # Title
        title = f"{ticker} Analysis (Hypothetical)"
        title_style.fontSize = 24  # Adjust font size for this title
        title_style.leading = 28  # Adjust leading for this title
        title_paragraph = Paragraph(title, title_style)

        # Subtitle
        subtitle = ("Combined portfolio returns matrix at expiration is modeled with the defined outcomes for the concentrated position and the broad market exposure.")
        subtitle_paragraph = Paragraph(subtitle, ParagraphStyle(name='Subtitle', fontName='sans_l', fontSize=14, leading=18, textColor=colors.HexColor('#4569A8')))

        # Right-side content
        right_content = ("• Concentrated position is fully hedged at pre-defined level.<br/>"
                         "<br/>"
                         "• Total equity market exposure remains unchanged.<br/>"
                         "<br/>"
                         "• Upside and downside risk is transferred from the single stock to the S&amp;P 500.")
        right_content_paragraph = Paragraph(right_content, table_header_style)

        date = dt.datetime.today().strftime('%m/%d/%Y')
        # Footer text
        footer_text = ("Note: Performance illustrated on this page is hypothetical in nature based on the price of the underlying position and market conditions at the time the report was generated. "
                       f"Strategy returns are illustrated gross of Lido’s management fees. Data source: Thomson Reuters, as of market close {date}.")
        footer_paragraph = Paragraph(footer_text, footer_style)

        # Add image
        matrix_image_path = Path(f"assets/images/heatmap_{ticker}.png")  # Update with the actual path
        matrix_image_width = 6.5 * inch
        matrix_image_height = 4.6 * inch
        matrix_image_x = 0.833 * inch
        matrix_image_y = 1.25 * inch

        # Draw elements on the canvas at specified positions
        title_y = (7.5 * inch) - (0.67 * inch)
        subtitle_y = (7.5 * inch) - (1.2 * inch)
        right_content_y = (7.5 * inch) - (2.2 * inch)
        footer_y = 7.5 * inch - 6.5 * inch

        # Draw elements
        title_paragraph.wrapOn(c, 12 * inch, inch)
        title_paragraph.drawOn(c, 0.333 * inch, title_y)

        subtitle_paragraph.wrapOn(c, 12 * inch, inch)
        subtitle_paragraph.drawOn(c, 0.333 * inch, subtitle_y)

        c.drawImage(matrix_image_path, matrix_image_x, matrix_image_y, width=matrix_image_width, height=matrix_image_height)

        right_content_paragraph.wrapOn(c, 4 * inch, inch)
        right_content_paragraph.drawOn(c, 8.5 * inch, 3.5 * inch)

        footer_paragraph.wrapOn(c, 13 * inch, inch)
        footer_paragraph.drawOn(c, 0.333 * inch, footer_y)

        return c

    def add_standards(self, c: canvas.Canvas, path):
        c = self._gen_content_page(c)
        c.drawImage(path, 0, 0, width=13.333 * inch, height=7.5 * inch,
                    mask='auto')  # 'mask="auto"' for transparency
        return c


    def create_csp_report(self):
    # Generate the reportd
        client_name = self.csp_client.client_name
        c = self._generate_csp_report()
        c = self._add_title_page(c)
        c = self.add_standards(c, toc)
        c = self.add_standards(c, firm_div)
        c = self.add_standards(c, about)
        c = self.add_standards(c, csp_div)
        c = self.add_standards(c, csp_over)
        c = self.add_standards(c, csp_over_tab)
        c = self._create_summary_analysis_page(c)
        c = self.add_standards(c, hedge_div)
        c = self.add_standards(c, hedge_over)
        c = self.create_cc_page(c)
        c = self.create_payoff_diagram_page(c)
        c = self.add_standards(c, cc_div)
        c = self.add_standards(c, cc_over)
        c = self.create_covered_calls_page(c)
        c = self.create_covcall_payoff_page(c)
        c = self.create_tax_neutral_page(c)
        c = self.add_standards(c, sef_div)
        c = self.add_standards(c, sef_over)
        c = self.add_standards(c, sef_proc)
        c = self.create_sef_strategy_page(c)
        c = self.create_stock_sef_page(c)
        c = self.create_sef_matrix_page(c)
        c = self.add_standards(c, disc_div)
        c = self.add_standards(c, notes)
        c = self.add_standards(c, defin)
        c = self.add_standards(c, discs)
        c.save()
        return

    def create_csp_report_ex_sef(self):
    # Generate the reportd
        client_name = self.csp_client.client_name
        c = self._generate_csp_report()
        c = self._add_title_page(c)
        c = self.add_standards(c, toc_ex)
        c = self.add_standards(c, firm_div)
        c = self.add_standards(c, about)
        c = self.add_standards(c, csp_div)
        c = self.add_standards(c, csp_over)
        c = self.add_standards(c, csp_over_tab)
        c = self._create_summary_analysis_page(c)
        c = self.add_standards(c, hedge_div)
        c = self.add_standards(c, hedge_over)
        c = self.create_cc_page(c)
        c = self.create_payoff_diagram_page(c)
        c = self.add_standards(c, cc_div)
        c = self.add_standards(c, cc_over)
        c = self.create_covered_calls_page(c)
        c = self.create_covcall_payoff_page(c)
        c = self.create_tax_neutral_page(c)
        c = self.add_standards(c, disc_div)
        c = self.add_standards(c, notes)
        c = self.add_standards(c, defin)
        c = self.add_standards(c, discs)
        c.save()
        return


