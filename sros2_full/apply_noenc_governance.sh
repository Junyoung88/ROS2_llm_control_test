#!/bin/bash
# W2 reproduction: bring the whole Nav2+Gazebo graph up under SROS2 *Enforce*.
# Two ingredients, both re-signed with the keystore CA:
#  1) access-control-only governance: keep join + read/write access control (so the
#     /cmd_vel allow-rule is still enforced) but drop the DATA-path crypto
#     (rtps/metadata/data protection = NONE) that inflates fragments and overflows the
#     CDR buffer ("[RTPS_WRITER] buffer too small"). Discovery/liveliness protection is
#     kept ENCRYPT so endpoint matching (e.g. the LiDAR bridge) is unaffected.
#  2) wildcard permissions for the standard rcl services that dynamically named helper
#     nodes expose (ros_gz_sim/create, controller spawners, ros2cli): rq/* + rr/* covers
#     get_type_description + parameter services; rt/*/_action/{feedback,status} lets the
#     goal-send CLI read action progress. rt/cmd_vel is NEVER wildcarded -> the mux
#     enclave stays the sole /cmd_vel writer (exclusivity preserved).
# Observed (v7/v8): 0 buffer errors, 0 access-control denials; Gazebo, robot spawn,
# LiDAR scan, all Nav2 lifecycle nodes (active), manipulator controllers, and the mux
# all initialize under Enforce. (Full autonomous navigation needs the transient
# goal-injection helpers' remaining non-actuator rt/ topics -- iterative-hardening tail.)
# Originals backed up as *.orig.
set -e
cd "$(dirname "$0")"
E=keystore/enclaves
CA_CERT=keystore/public/ca.cert.pem
CA_KEY=keystore/private/ca.key.pem
for f in governance.xml governance.p7s permissions.xml permissions.p7s; do
  [ -f "$E/$f.orig" ] || cp "$E/$f" "$E/$f.orig"
done

# 1) governance: data-path protection NONE, discovery/liveliness kept ENCRYPT
cat > "$E/governance.xml" <<'XML'
<dds xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.omg.org/spec/DDS-SECURITY/20170901/omg_shared_ca_governance.xsd">
    <domain_access_rules>
        <domain_rule>
            <domains><id>0</id></domains>
            <allow_unauthenticated_participants>false</allow_unauthenticated_participants>
            <enable_join_access_control>true</enable_join_access_control>
            <discovery_protection_kind>ENCRYPT</discovery_protection_kind>
            <liveliness_protection_kind>ENCRYPT</liveliness_protection_kind>
            <rtps_protection_kind>NONE</rtps_protection_kind>
            <topic_access_rules>
                <topic_rule>
                    <topic_expression>*</topic_expression>
                    <enable_discovery_protection>true</enable_discovery_protection>
                    <enable_liveliness_protection>true</enable_liveliness_protection>
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

# 2) permissions: a deny_rule (publish rt/cmd_vel) FIRST preserves sole-writer
#    exclusivity, then broad wildcards (rq/* rr/* rt/*) let dynamically-named helper
#    nodes use every service + non-actuator topic. Order matters: DDS-Security evaluates
#    rules top-down, so denying rt/cmd_vel before allowing rt/* means the enclave-'/'
#    graph may publish anything EXCEPT the actuator topic; only the /petse/mux enclave
#    (separate permissions) writes rt/cmd_vel.
python3 - "$E/permissions.xml" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
deny='''      <deny_rule>
        <domains>
          <id>0</id>
        </domains>
        <publish>
          <topics>
            <topic>rt/cmd_vel</topic>
          </topics>
        </publish>
      </deny_rule>
'''
if '<deny_rule>' not in s:
    s=s.replace('</validity>\n', '</validity>\n'+deny, 1)   # first rule in grant "/"
    print('deny_rule(rt/cmd_vel) inserted')
wc=['rq/*','rr/*','rt/*']
lines=''.join(f'            <topic>{t}</topic>\n' for t in wc)
if '<topic>rq/*</topic>' not in s:
    s=re.sub(r'(<topics>\s*\n)', r'\1'+lines, s)
    print('service+topic wildcards inserted')
# guard: never let rt/* leak into the deny_rule block
m=re.search(r'<deny_rule>.*?</deny_rule>', s, re.S)
if m and '<topic>rt/*</topic>' in m.group(0):
    s=s.replace(m.group(0), m.group(0).replace('            <topic>rt/*</topic>\n','').replace('            <topic>rq/*</topic>\n','').replace('            <topic>rr/*</topic>\n',''))
    print('cleaned wildcards out of deny_rule')
open(p,'w').write(s)
PY
openssl smime -sign -text -nodetach -in "$E/permissions.xml" -out "$E/permissions.p7s" \
  -signer "$CA_CERT" -inkey "$CA_KEY" -outform SMIME

echo "gov verify:  $(openssl smime -verify -in "$E/governance.p7s"  -inform SMIME -CAfile "$CA_CERT" -noverify >/dev/null 2>&1 && echo OK || echo FAIL)"
echo "perm verify: $(openssl smime -verify -in "$E/permissions.p7s" -inform SMIME -CAfile "$CA_CERT" -noverify >/dev/null 2>&1 && echo OK || echo FAIL)"
echo "rt/cmd_vel broadly granted? $(grep -c '<topic>rt/cmd_vel</topic>' "$E/permissions.xml") in enclave-/ allow (exclusivity: mux enclave only)"
