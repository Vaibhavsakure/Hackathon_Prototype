from report_generator import generate_batch_report
from deviation_engine import analyze_deviation
import pandas as pd

df = pd.read_csv('data/master_df.csv')

# Test with T001 (non-golden batch, has deviations)
report = analyze_deviation('T001', mode='balanced')
print("Report status:", report.get('overall_status'))
print("Top actions:", len(report.get('top_actions', [])))

batch_row = df[df['Batch_ID'] == 'T001']
batch_data = batch_row.iloc[0].to_dict() if not batch_row.empty else None

try:
    pdf_bytes = generate_batch_report(report, batch_data=batch_data, mode='balanced')
    print(f"PDF generated: {len(pdf_bytes)} bytes")
    
    with open('C:/tmp/test_t001.pdf', 'wb') as f:
        f.write(pdf_bytes)
    print("Saved to C:/tmp/test_t001.pdf")
    
    header = pdf_bytes[:10]
    print(f"First 10 bytes: {header}")
    
    page_count = pdf_bytes.count(b'/Type /Page')
    print(f"Estimated pages: {page_count}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
