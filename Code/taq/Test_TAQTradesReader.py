import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add Code directory

from taq import MyDirectories
from taq.TAQQuotesReader import TAQQuotesReader
from taq.TAQTradesReader import TAQTradesReader


class Test_TAQTradesReader(unittest.TestCase):

    def test1(self):

        reader = TAQTradesReader( MyDirectories.getTradesDir() + '/20070920/IBM_trades.binRT' )
        
        zz = list([
            reader.getN(),
            reader.getSecsFromEpocToMidn(),
            reader.getMillisFromMidn( 0 ),
            reader.getSize( 0 ),
            reader.getPrice( 0 )
        ])

        self.assertEqual(
            '[25367, 1190260800, 34210000, 76600, 116.2699966430664]',
            str( zz )
        )


if __name__ == "__main__":
    unittest.main()
