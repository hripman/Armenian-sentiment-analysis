# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sys

CONSUMER_KEY = 'NgmQimyBVaW89ubwlRKoPt6Uf'
CONSUMER_SECRET = 'qrNYgDckElyF4yqoMdIiaqrT0XW2y5WyLRo6FpN1RVPfYTQ6R1'

ACCESS_TOKEN_KEY = '934419909114724352-8M1rr24LxcC2CUZnJZpcQgv2qBy2RL0'
ACCESS_TOKEN_SECRET = 'qWYEjpHLcXIzXrj6tuFNDCUXPFvfBH1RhTqC07K4UICB6'

if CONSUMER_KEY is None or CONSUMER_SECRET is None or ACCESS_TOKEN_KEY is None or ACCESS_TOKEN_SECRET is None:
    print("""\
When doing last code sanity checks for the book, Twitter
was using the API 1.0, which did not require authentication.
With its switch to version 1.1, this has now changed.

It seems that you don't have already created your personal Twitter
access keys and tokens. Please do so at
https://dev.twitter.com/docs/auth/tokens-devtwittercom
and paste the keys/secrets into twitterauth.py

Sorry for the inconvenience,
The authors.""")

    sys.exit(1)
