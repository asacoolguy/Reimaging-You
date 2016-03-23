import MySQLdb
import gc

db = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="localroot", db="test")
cursor = db.cursor()

#cursor.execute("SELECT feat FROM ethan.feat$1gram$statuses_ethan$user_id$16to16;", feat)
cursor.execute("SELECT * FROM ethan_status")

results = cursor.fetchall()
i = 0

for row in results:
  p_id = row[0]
  word = row[2]
  frequency = row[3]
  # Now print fetched result
  print "id = %s, word = %s, frequency = %d" % \
         (p_id, word, frequency)
  i = i + 1
  if i > 4:
  	break



db.close()
gc.collect()