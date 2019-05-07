"""
Import the client certificate and the private key
"""

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"

import os
from tlslite import X509, X509CertChain, parsePEMKey

cert_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clientX509cert.crt")
file = open(cert_file_path).read()
x509 = X509()
x509.parse(file)
cert_chain = X509CertChain([x509])

key_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clientX509key.key")
file = open(key_file_path).read()
private_key = parsePEMKey(file, private=True)