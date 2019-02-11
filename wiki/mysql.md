# mysql IDEAS

## grant user privileges
```
GRANT ALL PRIVILEGES ON db_name.* TO 'username'@'localhost' IDENTIFIED BY 'password';

SHOW GRANTS FOR 'username'@'localhost';

examples:
GRANT ALL PRIVILEGES ON global_house.* TO 'hu'@'localhost' IDENTIFIED BY 'hu123';
```

```
+--------------------------------------------------------------+
| Grants for hu@localhost                                      |
+--------------------------------------------------------------+
| GRANT USAGE ON *.* TO 'hu'@'localhost'                       |
| GRANT ALL PRIVILEGES ON `global_house`.* TO 'hu'@'localhost' |
+--------------------------------------------------------------+
```

## create table
```
CREATE TABLE table_name (
  column_name1 definition,
  column_name1 definition,
  column_name1 definition,
  options
);

EX:

CREATE TABLE subjects (
  id INT(11) NOT NULL AUTO_INCREMENT,
  menu_name VARCHAR(255),
  position INT(3),
  visible TINYINT(1),
  PRIMARY KEY (id)
  );
```

```
use database_name;

CREATE TABLE subjects (
  id INT(11) NOT NULL AUTO_INCREMENT,
  menu_name VARCHAR(255),
  position INT(3),
  visible TINYINT(1),
  PRIMARY KEY (id)
  );

SHOW TABLES;

SHOW COLUMNS FROM subjects;

DROP TABLE table_name;
```

## insert columns
after create field for table, insert data into table:
```
INSERT INTO table_name (col1, col2, col3) VALUES (val1, val2, val3);

INSERT INTO subjects (menu_name, position, visible) VALUES ('About', 1, 1);
```

## read data from table
```
SELECT * FROM table_name WHERE column1='some_text' ORDER BY col1 ASC;

SELECT * FROM subjects;
```

## update table data
```
UPDATE table_name SET col1='this', col2='that' WHERE id = 1;
UPDATE subjects SET menu_name='windows', position=4 WHERE id = 3;
```

## delete table data
```
DELETE FROM table where id = 1;

DELETE FROM subjects where id=3;
```

## basic processing
```
<?php
echo "hello <br />";
$name = $_GET['q'] ?? " ";
echo $name;

$connect = mysqli_connect('localhost', 'hu', 'hu123', 'global_house');
if (!$connect) {
  die('could not connect : '.mysqli_error($connect));
}

mysqli_select_db($connect, 'subjects');
$sql = "select * from subjects where id=".$name;
$result = mysqli_query($connect, $sql);

//var_dump($result);
echo '<table>';
echo  "<thead>
  <tr>
    <th>id</th>
    <th>name</th>
    <th>pos</th>
    <th>visible</th>
  </tr>  </thead>";
  while ($row = mysqli_fetch_array($result)) {
    echo '<tbody><tr>';
    echo "<td>" . $row['id']."</td>";
    echo "<td>" . $row['menu_name']."</td>";
    echo "<td>" . $row['position']."</td>";
    echo "<td>" . $row['visible']."</td>";
    echo "</tr></tbody>";
  }
echo "</table>"
?>
```

explanation:
```
1. php opens a connection to a mysql server
2. the correct person is found;
3. and html table is created, filled with data, and send back to the txthint placeholder.
```


## typical code blocks
1. mysqli_fetch_assoc()

```
$result = find_all_subjects();

while ($subject = mysqli_fetch_assoc($result)) {
  echo $subject['menu_name'];
}
```

2. quote
sql query alway be double quote, then you can use the single quote inside double quote.

```
$sql = "WHERE id='" . "';";
```
3. batch load cmd from \*.sql file
```
bash: mysql -u root -p < *.sql
```
then you will get

## error handling
mysqli_connect_errono() return number.
mysqli_connect_error() return string.

## 5 steps for php and databases interaction
1. Create a database connection
2. Perform a database query
3. Use returned data(if any)
4. Release returned data
5. Close the database connection.

```
$db = mysqli_connect(DB_SERVER, DB_USER, DB_PASS, DB_NAME);

$sql = "SELECT * FROM subjects";
$sql .= "ORDER BY position ASC";
$subject_set = mysqli_query($db, $sql);

while($subject = mysqli_fetch_assoc($subject_set)) {
  echo $subject['menu_name'];
}

mysqli_free_result($subject_set);

mysqli_close($db);
```

## Tips
1. check charset
```
show create table book_main_category \G;
```
