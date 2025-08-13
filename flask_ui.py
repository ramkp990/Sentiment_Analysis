import logging
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file
from io import BytesIO

app = Flask(__name__)

# Global dictionary to store the extracted data
extracted_data = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    keyword1 = request.form.get('keyword1')
    keyword2 = request.form.get('keyword2')
    logging.info(f"Searching for keywords: {keyword1}, {keyword2}")

    # Find rows containing either keyword
    results = []
    for data_id, data in extracted_data.items():
        if any(keyword1.lower() in element.lower() or keyword2.lower() in element.lower() for element in data['keywords']):
            results.append(data)

    # Calculate sentiment percentages
    pos_percentage1, pos_percentage2 = calculate_sentiment_percentages(results, keyword1, keyword2)

    return render_template('results.html', keyword1=keyword1, keyword2=keyword2,
                           pos_percentage1=pos_percentage1, pos_percentage2=pos_percentage2)

@app.route('/update_data', methods=['POST'])
def update_data():
    data = request.json
    data_id = data['id']
    extracted_data[data_id] = data['data']
    return jsonify(success=True)

@app.route('/graph/<keyword1>/<keyword2>')
def graph(keyword1, keyword2):
    data_keyword1 = [
        data for data in extracted_data.values() if keyword1 in data['keywords']
    ]
    data_keyword2 = [
        data for data in extracted_data.values() if keyword2 in data['keywords']
    ]

    df1 = pd.DataFrame(data_keyword1)
    df2 = pd.DataFrame(data_keyword2)

    if df1.empty and df2.empty:
        return "No data available for these keywords"

    def parse_date(date_str):
        for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d'):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                pass
        raise ValueError(f"No valid date format found for {date_str}")

    if not df1.empty:
        df1['utc_created'] = df1['utc_created'].apply(parse_date)
        df1['sentiment'] = df1['sentiment'].str.lower().map({'positive': 1, 'negative': 0})

    if not df2.empty:
        df2['utc_created'] = df2['utc_created'].apply(parse_date)
        df2['sentiment'] = df2['sentiment'].str.lower().map({'positive': 1, 'negative': 0})

    plt.figure(figsize=(10, 6))

    if not df1.empty:
        plt.scatter(df1['utc_created'], df1['sentiment'], marker='o', c='b', label=keyword1)

    if not df2.empty:
        plt.scatter(df2['utc_created'], df2['sentiment'], marker='o', c='r', label=keyword2)

    plt.yticks([0, 1], labels=['Negative', 'Positive'])
    plt.xlabel('UTC Created')
    plt.ylabel('Sentiment')
    plt.title(f'Sentiment over Time for Keywords: {keyword1} (blue) and {keyword2} (red)')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

def calculate_sentiment_percentages(results, keyword1, keyword2):
    count1_pos, count1_total = 0, 0
    count2_pos, count2_total = 0, 0

    for result in results:
        if keyword1 in result['keywords']:
            count1_total += 1
            if result['sentiment'].lower() == 'positive':
                count1_pos += 1
        if keyword2 in result['keywords']:
            count2_total += 1
            if result['sentiment'].lower() == 'positive':
                count2_pos += 1

    pos_percentage1 = (count1_pos / count1_total * 100) if count1_total > 0 else 0
    pos_percentage2 = (count2_pos / count2_total * 100) if count2_total > 0 else 0

    return pos_percentage1, pos_percentage2

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5001)
