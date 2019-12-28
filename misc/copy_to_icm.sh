#!/bin/bash
# Example script to copy things necessary to compute results for protocols to icm
# Assuming that we can connect to icm using "ssh icm"

ssh icm "mkdir protocols"
scp -r resources/protocols/data/protocol_times_*.csv icm:~/protocols/
scp run.sl build/tumor resources/tumors/out-vnw-tr1-st0-0a-initial.json icm:~/