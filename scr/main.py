from vba import VectorBasedAlgorithm as vba
from hba import HonestyBasedAlgorithm as hba
from kba import KLDBasedAlgorithm as kba
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

    # KBA algoritmin implementaatio
    kba_anomalies = kba.detect_anomalies()