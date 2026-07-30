#!/bin/bash
# W2 reproduction: convert the SROS2 keystore to an ACCESS-CONTROL-ONLY governance
# (join + read/write access control kept, all protection_kind=NONE) and add a wildcard
# get_type_description permission for transient ros2cli participants. This removes the
# crypto-transform fragment overflow ("[RTPS_WRITER] buffer too small") that stalled
# whole-graph Enforce, while still enforcing the /cmd_vel allow-rule (sole-writer).
# Result observed: 0 "buffer too small", 0 access-control denials; all 14 Nav2 nodes
# authenticate under Enforce. (A full navigating trial additionally needs enclaves for
# the Gazebo bridge/sensor nodes.) Originals are backed up as *.orig.
set -e
cd "$(dirname "$0")"
E=keystore/enclaves
CA_CERT=keystore/public/ca.cert.pem
CA_KEY=keystore/private/ca.key.pem

[ -f "$E/governance.xml.orig" ]  || cp "$E/governance.xml"  "$E/governance.xml.orig"
[ -f "$E/governance.p7s.orig" ]  || cp "$E/governance.p7s"  "$E/governance.p7s.orig"
[ -f "$E/permissions.xml.orig" ] || cp "$E/permissions.xml" "$E/permissions.xml.orig"
[ -f "$E/permissions.p7s.orig" ] || cp "$E/permissions.p7s" "$E/permissions.p7s.orig"

# 1) access-control-only governance (protection kinds NONE, access control retained)
cat > "$E/governance.xml" <<'XML'
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_governance.xsd">
    <domain_access_rules>
        <domain_rule>
            <domains><id>0</id></domains>
            <allow_unauthenticated_participants>false</allow_unauthenticated_participants>
            <enable_join_access_control>true</enable_join_access_control>
            <discovery_protection_kind>NONE</discovery_protection_kind>
            <liveliness_protection_kind>NONE</liveliness_protection_kind>
            <rtps_protection_kind>NONE</rtps_protection_kind>
            <topic_access_rules>
                <topic_rule>
                    <topic_expression>*</topic_expression>
                    <enable_discovery_protection>false</enable_discovery_protection>
                    <enable_liveliness_protection>false</enable_liveliness_protection>
                    <enable_read_access_control>true</enable_read_access_control>
                    <enable_write_access_control>true</enable_write_access_control>
                    <metadata_protection_kind>NONE</metadata_protection_kind>
                    <data_protection_kind>NONE</data_protection_kind>
                </topic_rule>
            </topic_access_rules>
        </domain_rule>
    </domain_access_rules>
</dds>
XML
openssl smime -sign -text -nodetach -in "$E/governance.xml" -out "$E/governance.p7s" \
  -signer "$CA_CERT" -inkey "$CA_KEY" -outform SMIME

# 2) add wildcard get_type_description topics for dynamically named ros2cli participants
python3 - "$E/permissions.xml" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
ins=('            <topic>rq/*/get_type_descriptionRequest</topic>\n'
     '            <topic>rr/*/get_type_descriptionReply</topic>\n')
open(p,'w').write(re.sub(r'(<topics>\s*\n)', r'\1'+ins, s))
PY
openssl smime -sign -text -nodetach -in "$E/permissions.xml" -out "$E/permissions.p7s" \
  -signer "$CA_CERT" -inkey "$CA_KEY" -outform SMIME

echo "no-encryption governance + wildcard permissions applied and signed."
echo "verify governance:  $(openssl smime -verify -in "$E/governance.p7s"  -inform SMIME -CAfile "$CA_CERT" -noverify >/dev/null 2>&1 && echo OK || echo FAIL)"
echo "verify permissions: $(openssl smime -verify -in "$E/permissions.p7s" -inform SMIME -CAfile "$CA_CERT" -noverify >/dev/null 2>&1 && echo OK || echo FAIL)"
