# python -m http.server --cgi

import cgi
import predict

form = cgi.FieldStorage()

review = form.getvalue('review')
pred = predict.test(review)

if pred == 0:
    msg = "Negative"
else:
    msg = "Positive"

print("""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>

<body>

<h1>Sentiment is {}</h1>

</body>
</html>
""".format(msg))
