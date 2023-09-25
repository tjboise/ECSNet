import smtplib
from email.mime.text import MIMEText

def sendMail(message,Subject,to_addrs='tizhan1234@gmail.com'):
    '''
    :param message: str 邮件内容
    :param Subject: str 邮件主题描述
    :param sender_show: str 发件人显示，不起实际作用如："xxx"
    :param recipient_show: str 收件人显示，不起实际作用 多个收件人用','隔开如："xxx,xxxx"
    :param to_addrs: str 实际收件人
    :param cc_show: str 抄送人显示，不起实际作用，多个抄送人用','隔开如："xxx,xxxx"
    '''
    # 填写真实的发邮件服务器用户名、密码
    user = 'tjzhang_testbox1@outlook.com'
    password = 'Leke123123'
    # 邮件内容
    msg = MIMEText(message, 'plain', _charset="utf-8")
    # 邮件主题描述
    msg["Subject"] = Subject
    server = smtplib.SMTP("smtp.office365.com", 587)
    server.starttls()

    server.login('tjzhang_testbox1@outlook.com', 'Leke123123')

    server.sendmail(
        from_addr='tjzhang_testbox1@outlook.com',
        to_addrs = to_addrs,
        msg=msg.as_string())

    print('success!')




if __name__ == "__main__":
    # 邮件内容
    message = 'Hi Tianjie, \n This is message to inform you that your training in the lab is already done!'
    subject = 'TJ, Your training is done! for today'
    sendMail(message,subject)





