import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse


def send_email(to_email, from_email, smtp_server, smtp_port, smtp_username, smtp_password, subject, message):
    # 创建一个带附件的 email 消息实例
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # 邮件正文内容
    msg.attach(MIMEText(message, 'plain'))

    # 连接到邮件服务器
    try:
        # server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.login(smtp_username, smtp_password)

        # 发送邮件
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='发送邮件')
    # parser.add_argument('input_file', help='Input file to process')
    parser.add_argument('-d', '--domail', help='域名')
    parser.add_argument('-t', '--to_email', help='发送到的Email地址')
    parser.add_argument('-f', '--from_email', help='发送者邮箱')
    parser.add_argument('-p', '--passwd', help='发送者密码')
    parser.add_argument('-s', '--subject', help='主题')
    parser.add_argument('-c', '--content', help='内容')
    args = parser.parse_args()
    '''
    with open(args.input_file, 'r') as f:
        data = f.read()
    '''

    domail = args.domail
    send_email(
        to_email=args.to_email,
        from_email=args.from_email,
        smtp_server=f"smtp.{domail}",
        smtp_port=25,
        smtp_username=args.from_email,
        smtp_password=args.passwd,
        subject=args.subject,
        message=args.content
    )
