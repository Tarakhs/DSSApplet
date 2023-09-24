To build docker:

1) cd webapp

2) sudo docker build -t dss_webapp .

3) sudo docker run -p 8501:8501 dss_webapp (Must have port 8501 available)

Navigate to 127.0.0.1:8501

Data inserted must be of csv format with last row strictly labeled as "Classification"
