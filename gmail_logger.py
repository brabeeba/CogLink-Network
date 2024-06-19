import logging
import logging.handlers
import json
 
class GmailLogger(logging.handlers.SMTPHandler):
	def emit(self, record):
		try:
			import smtplib
			import string
			try:
				from email.utils import formatdate
			except ImportError:
				formatdate = self.date_time
			port = self.mailport
			if not port:
				port = smtplib.SMTP_PORT
			smtp = smtplib.SMTP(self.mailhost, port)
			msg = self.format(record)
			msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\nDate: %s\r\n\r\n%s" % (
							self.fromaddr,
							string.join(self.toaddrs, ","),
							self.getSubject(record),
							formatdate(), msg)
			if self.username:
				smtp.ehlo() # for tls add this line
				smtp.starttls() # for tls add this line
				smtp.ehlo() # for tls add this line
				smtp.login(self.username, self.password)
			smtp.sendmail(self.fromaddr, self.toaddrs, msg)
			print "sent"
			smtp.quit()
		except:
			self.handleError(record)
 
# logger = logging.getLogger()

# with open('credential.json', 'r') as f:
# 	info = json.load(f)
# 	account, password = info['account'], info['password']
 
# gm = GmailLogger(("smtp.gmail.com", 587), 'secretsanta2015harvard@gmail.com', ['mienwang@college.harvard.edu'], 'test', (account, password))
# gm.setLevel(logging.INFO)
 
# logger.addHandler(gm)
# logger.warning('Test')
