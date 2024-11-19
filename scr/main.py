from vba import VectorBasedAlgorithm as vba
from hba import HonestyBasedAlgorithm as hba
def main():
    """
    TODO
    Etsi hyvä datasetti
    Muokkaa datasetti muotoon DF[meters, readings]
    Ja kovaa ajoa
    """

    # VBA algoritmin implementaatio
    vba_anomalies = vba.detect_anomalies()


    # HBA algoritmin implementaatio
    hba_anomalies = hba.detect_anomalies()
    