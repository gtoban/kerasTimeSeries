import smtplib, ssl
import subprocess
import time
import sys


def sendEmail(password):
    port =  465
    
    email = "gtoban87@gmail.com"
    message = "Subject: Job Complete\n\n\n\nThe job is complete"

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com",port,context=context) as server:
        server.login(email, password)
        server.sendmail(email,email,message)


def main():
    password = input("Enter password: ")
    
    sendEmail(password)
    

main()
