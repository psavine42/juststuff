
"""
For a LinkType - Generate functions

if the link is not 1-to-1, procedures must exist for common cases
Join 
"""
class LinkType(object):
    def __init__(self, from_table, to_table,
                 ids_from, ids_to
                 ):
        self._from = from_table
        self._to = to_table

    def store_fn(self):
        params = self.lookup_cols(self._to)

        sql = """
        CREATE FUNCTION {name} (
            {params}
        )
        RETURNS TABLE
        AS
        RETURN
            {do};    
        """.format(name=self.name(), params=params, do='')

    def procedure(self):
        """
        CREATE PROCEDURE SelectAllCustomers @City nvarchar(30)
        AS
        SELECT * FROM Customers WHERE City = @City
        GO;
        """

    def join_fn(self):
        """

        """

    def create_view(self):
        """
        create the linked items as a view
        """
        return

    def name(self):
        return "{}_{}".format(self._from, self._to)

    def lookup_cols(self, name):
        return []

