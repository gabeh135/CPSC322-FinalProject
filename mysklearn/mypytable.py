"""
Programmer: Gabe Hoing
Class: CPSC 322, Fall 2024
Programming Assignment #7
11/20/2024
I attempted all bonus questions.

Description: This program creates a Python representation of a 2D table.
Includes various table methods such as joins and pretty print.
"""
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def print_number_instances(self, table_name):
        """Prints the number of instances in the table.

        Args:
            table_name(str): title of the table
        """
        print(f"{table_name} has {len(self.data)} instances")

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_frequencies(self, col_name):
        """Returns the frequencies of column values in the table for
            the specified column.

        Args:
            col_name(str): name of the column containing wanted data

        Returns:
            list: the values of unique columns
            list: the respective counts to unique_col_values
        """
        col = self.get_column(col_name)
        unique_col_values = sorted(list(set(col)))

        counts = []
        for val in unique_col_values:
            counts.append(col.count(val))

        return unique_col_values, counts

    def get_index(self, column_name):
        """Computes the index of a given column name.

        Args:
            column_name(str): wanted column
        Returns:
            int: index of column
        """
        return self.column_names.index(column_name)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.get_index(col_identifier)

        col = []
        for row in self.data:
            value = row[col_index]
            if include_missing_values or value != "NA":
                col.append(value)

        return col

    def get_columns(self, col_identifiers, include_missing_values=True):
        """Extracts columns from the table data as a list.

        Args:
            col_identifiers(list of str or int): list of strings for a 
                column name or ints for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: sublist of values in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_indices = [self.get_index(col_identifier) for col_identifier in col_identifiers]

        cols = []
        for row in self.data:
            values = [row[col_index] for col_index in col_indices]
            if include_missing_values or "NA" not in values:
                cols.append(values)

        return cols

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, value in enumerate(row):
                try:
                    self.data[i][j] = float(value)
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        indexes = sorted(row_indexes_to_drop, reverse=True)
        for index in indexes:
            self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        with open(filename, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            for row in reader:
                table.append(row)

        self.column_names = table[0]
        self.data = table[1:]

        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        table = self.data.copy()
        table.insert(0, self.column_names)

        with open(filename, "w", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(table)
            outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        rows = set()
        duplicate_indexes = []
        for i, row in enumerate(self.data):
            new_row = []
            for key in key_column_names:
                new_row.append(row[self.get_index(key)])
            row_tuple = tuple(new_row)
            if row_tuple in rows:
                duplicate_indexes.append(i)
            else:
                rows.add(row_tuple)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows = []
        for i, row in enumerate(self.data):
            for value in row:
                if value == "NA":
                    rows.append(i)
                    break
        self.drop_rows(rows)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        index = self.get_index(col_name)

        col_sum, count = 0, 0
        for row in self.data:
            value = row[index]
            if value != "NA":
                col_sum += value
                count += 1

        average = col_sum / count
        for row in self.data:
            if row[index] == "NA":
                row[index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        table = []
        for col_name in col_names:
            col_index = self.get_index(col_name)
            column = [row[col_index] for row in self.data if row[col_index] != "NA"]

            column.sort()

            col_length = len(column)

            if col_length > 0:
                minimum = column[0]
                maximum = column[-1]
                mid = (minimum + maximum) / 2
                average = sum(column) / col_length

                med_index = (col_length - 1) // 2
                if col_length % 2:
                    med = column[med_index]
                else:
                    med = (column[med_index] + column[med_index + 1]) / 2

                table.append([col_name, minimum, maximum, mid, average, med])

        return MyPyTable(header, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        header = self.column_names + [col for col in other_table.column_names if col not in key_column_names]

        table = []
        for row1 in self.data:
            for row2 in other_table.data:
                is_match = True
                for key in key_column_names:
                    if row1[self.get_index(key)] != row2[other_table.get_index(key)]:
                        is_match = False
                        break
                if is_match:
                    row = []
                    for column in header:
                        if column in self.column_names:
                            row.append(row1[self.get_index(column)])
                        else:
                            row.append(row2[other_table.get_index(column)])
                    table.append(row)

        return MyPyTable(header, table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        def get_row(row, table, col_names):
            """Helper function to retrieve a row from a table based on column names, filling missing values with "NA".

            Args:
                row(list): row to get values from.
                table(MyPyTable): table this row is a part of.
                col_names(list of str): list of column names to extract.
            Returns:
                list: extracted row data.
            """
            return [row[table.get_index(column)] if column in table.column_names else "NA" for column in col_names]

        def match_rows(row1, row2):
            """Helper function to check whether two rows match based on the key columns.

            Args:
                row1(list): row from this table.
                row2(list): row from other_table.

            Returns:
                boolean: whether the rows match.
            """
            for key in key_column_names:
                if row1[self.get_index(key)] != row2[other_table.get_index(key)]:
                    return False
            return True

        header = self.column_names + [column for column in other_table.column_names if column not in key_column_names]

        table = []
        for row1 in self.data:
            matched = False
            for row2 in other_table.data:
                if match_rows(row1, row2):
                    matched = True
                    row = []
                    for column in header:
                        if column in self.column_names:
                            row.append(row1[self.get_index(column)])
                        else:
                            row.append(row2[other_table.get_index(column)])
                    table.append(row)

            if not matched:
                table.append(get_row(row1, self, header))

        for row2 in other_table.data:
            matched = False
            for row1 in self.data:
                if match_rows(row1, row2):
                    matched = True
                    break

            if not matched:
                table.append(get_row(row2, other_table, header))

        return MyPyTable(header, table)
