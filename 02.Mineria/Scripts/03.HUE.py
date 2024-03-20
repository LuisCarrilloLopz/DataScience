import pandas as pd
import calendar
import pyodbc
import psycopg2
import io
from typing import Optional, Iterator
from collections import deque

"""Conexión con Hue"""
def python_supra_conection(env, service):
    print("HACIENDO CONEXION")
    service = service.lower().strip()
    env = env.lower().strip()
    environment = {"pro": {"impala": "corporacionimpala.aacc.gs.corp", "hive": "hive2supra.unix.aacc.corp"},
                   "pre": {"impala": "impaladaemonsupre.unix.aacc.corp", "hive": "hiveserver2suprapre.unix.aacc.corp"}
                   }
    realm = {"pro": "IDMPRO.UNIX.AACC.CORP", "pre": "SANHQ.SANTANDERHQ.CORP"}
    cnxn = None
    if service == 'impala':
        connection_string_IaaS = f'''DRIVER=Cloudera ODBC Driver for Impala;Host={environment[env][service]};Port=21050;AuthMech=1;
                                     KrbRealm={realm[env]};KrbFQDN={environment[env][service]};KrbServiceName=impala;
                                     TrustedCerts=C:\Program Files (x86)\Cloudera ODBC Driver for Impala\lib\cacerts.pem;'''
        #        connection_string_IaaS  = f'''DSN=Cloudera ODBC Driver for Impala2;'''
        try:
            cnxn = pyodbc.connect(connection_string_IaaS, autocommit=True)
            print("CONEXION LISTA")
            return cnxn
        except:
            print("Error! Make sure that you have the correct Kerberos Ticket and Impala Driver")
            return cnxn
    elif service == 'hive':
        connection_string_IaaS = f'''DRIVER=Microsoft Hive ODBC Driver; Host={environment[env][service]};Port=10000;AuthMech=1;
                                     KrbRealm={realm[env]};KrbHostFQDN={environment[env][service]};KrbServiceName=hive;'''
        try:
            cnxn = pyodbc.connect(connection_string_IaaS, autocommit=True)
            print("CONEXION LISTA")
            return cnxn
        except:
            print("Error! Make sure that you have the correct Kerberos Ticket and Hive Driver")
            return cnxn
    else:
        print("Incorrect service!")
        return cnxn

"""Bucle para generar fechas"""
def generar_fechas_ultimo_dia_mes(year_ini,year_fin):
    fechas = []
    for year in range(year_ini, year_fin):  # Desde 2023 hasta 2024
        for mes in range(1, 13):  # Para cada mes del 1 al 12
            ultimo_dia = calendar.monthrange(year, mes)[1]
            fecha = f"{year}-{mes:02d}-{ultimo_dia:02d}"  # Formato de cadena 'AAAA-MM-DD'
            fechas.append(fecha)
    return fechas

"""Funcion para la Query__"""
def ejecutar_consulta_sql(fecha, schema, divisa, pais, aaaamm, cnxn):
    perimetro = f"""
create table default.perimetro_{pais}_{aaaamm} as
with saldos as
    (
        select A.*
            , case 
                when
                    l4 not in ('1840') and 
                    l5 not in ('12201','17320','17323') and 
                    l6 not in ('122113','122213','122313','124010','124113','124213','124313','173210','173220','173240','173252','173290') and 
                    l7 not in ('1224010','1224110','1244010','1244110') and 
                    (
                        l3 in ('122','124','184') or
                        l4 in ('1732') or
                        l5 in ('17981')
                    )
                    then importh
                else 0
            end as S10
            , case 
                when
                    l4 in ('1032','1034') or
                    l5 in ('12201') or
                    l6 in ('122113','122213','122313','124010','124113','124213','124313') or
                    l7 in ('1224010','1224110','1244010','1244110')
                    then importh
                else 0
            end as S11
            , case 
                when
                    l3 not in ('304') and 
                    l4 not in ('3022','3021') and 
                    l5 not in ('30002','30012','30022','30032','30302','30392') and 
                    l2 in ('30')
                    then importh
                else 0
            end as S12
            , case 
                when
                    l3 not in ('317') and 
                    l4 not in ('3101','3103','3111','3131','3141','3191') and 
                    l5 not in ('31500','31502','31511','31515') and 
                    l2 in ('31')
                    then importh
                else 0
            end as S13
            , case 
                when
                    l3 in ('101','102','112') or
                    l4 in ('1840') or
                    l5 in ('17320','17323') or
                    l6 in ('173210','173220','173240','173252','173290')
                    then importh
                else 0
            end as S14
            , case 
                when
                    l4 in ('3021','3022','3101','3103','3111','3131','3141','3191') or
                    l5 in ('30002','30012','30022','30032','30302','30392','31500','31502','31511','31515')
                    then importh
                else 0
            end as S15
            , case 
                when
                    l4 in ('1302','1311','1305','1314') or
                    l7 in ('1797210')
                    then importh
                else 0
            end as S16
            , case 
                when
                    l4 not in ('1401') and 
                    l5 not in ('14022','14032','14042','14051','14500') and 
                    l6 not in ('140010','140213','140313','140413','143010','143213','143313','143413') and 
                    l7 not in ('1405010','1405012','1435010') and 
                    (
                        l3 in ('140') or
                        l3 in ('143') or
                        l3 in ('145')
                    )
                    then importh
                else 0
            end as S30
            , case 
                when
                    l4 in ('1461','1463') or
                    l6 in ('140010','140213','140313','140413','143010','143213','143313','143413') or
                    l7 in ('1405010','1435010')
                    then importh
                else 0
            end as S31
            , case 
                when
                    l5 not in ('30402','30412','30422','30435','30436','30442','30446') and 
                    l3 in ('304')
                    then importh
                else 0
            end as S32
            , case 
                when
                    l5 not in ('31702','31706','31712','31716') and 
                    l3 in ('317')
                    then importh
                else 0
            end as S33
            , case 
                when
                    l5 not in ('14012') and 
                    l6 not in ('140512') and 
                    (
                        l4 in ('1401') or
                        l5 in ('14051','14500')
                    )
                    then importh
                else 0
            end as S34
            , case 
                when
                    l5 in ('30402','30412','30422','30435','30436','30442','30446','31702','31706','31712','31716')
                    then importh
                else 0
            end as S35
            , case 
                when
                    l4 in ('1410','1412')
                    then importh
                else 0
            end as S36
            , case 
                when
                    l5 in ('14022','14032','14042') or
                    l7 in ('1405012')
                    then importh
                else 0
            end as S40
            , case 
                when
                    l5 in ('14012') or
                    l6 in ('140512')
                    then importh
                else 0
            end as S43
            , case 
                when
                    l6 not in ('173101','173110') and 
                    (
                        l3 in ('121','123') or
                        l4 in ('1731','1031','1033') or
                        l7 in ('1797201','1797221')
                    )
                    then importh
                else 0
            end as S51
            , case 
                when
                    l5 not in ('11100','11110','11104','11114') and 
                    (
                        l3 in ('111') or
                        l4 in ('1131','1132','1133','1135') or
                        l6 in ('173101','173110') or
                        l7 in ('1797200','1797220')
                    )
                    then importh
                else 0
            end as S52
            , case 
                when
                    l4 in ('1304','1313','1306','1315') or
                    l7 in ('1797202') or
                    l8 in ('17972220','17972221')
                    then importh
                else 0
            end as S53
            , case 
                when
                    l4 not in ('1421','1441') and 
                    l7 not in ('1425012') and 
                    (
                        l3 in ('142') or
                        l3 in ('144') or
                        l4 in ('1460') or
                        l4 in ('1462')
                    )
                    then importh
                else 0
            end as S54
            , case 
                when
                    l5 not in ('14410') and 
                    (
                        l4 in ('1421','1441')
                    )
                    then importh
                else 0
            end as S55
            , case 
                when
                    l4 in ('1413')
                    then importh
                else 0
            end as S56
            , case 
                when
                    l7 in ('1425012')
                    then importh
                else 0
            end as S57
            , case 
                when
                    l6 not in ('173001','173010') and 
                    (
                        l3 in ('120') or
                        l4 in ('1730','1030')
                    )
                    then importh
                else 0
            end as S61
            , case 
                when
                    l4 in ('1603','1613')
                    then importh
                else 0
            end as ARS
            , case 
                when
                    l5 not in ('26460') and 
                    l6 not in ('264014','264031') and 
                    l7 not in ('2640301') and 
                    l8 not in ('26401102') and 
                    l9 not in ('264030010','264030213','264030313','264030413','264030610') and 
                    (
                        l4 in ('2646') or
                        l5 in ('26401','26403','26405')
                    )
                    then importh
                else 0
            end as F10
            , case 
                when
                    l9 in ('264030010','264030213','264030313','264030413','264030610')
                    then importh
                else 0
            end as F11
            , case 
                when
                    l4 in ('2621','2623')
                    then importh
                else 0
            end as F12
            , case 
                when
                    l4 in ('2631','2633')
                    then importh
                else 0
            end as F13
            , case 
                when
                    l6 in ('264031')
                    then importh
                else 0
            end as F14
            , case 
                when
                    l6 not in ('264041') and 
                    l7 not in ('2640401') and 
                    l9 not in ('264040010','264040213','264040313','264040413','264040610') and 
                    (
                        l5 in ('26404','26406')
                    )
                    then importh
                else 0
            end as F20
            , case 
                when
                    l9 in ('264040010','264040213','264040313','264040413','264040610')
                    then importh
                else 0
            end as F21
            , case 
                when
                    l4 in ('2624')
                    then importh
                else 0
            end as F22
            , case 
                when
                    l4 in ('2634')
                    then importh
                else 0
            end as F23
            , case 
                when
                    l6 in ('264041')
                    then importh
                else 0
            end as F24
            , case 
                when
                    l6 not in ('264002') and 
                    l7 not in ('2640001') and 
                    l9 not in ('264000010','264000213','264000313','264000413','264000610') and 
                    (
                        l5 in ('26400','26407')
                    )
                    then importh
                else 0
            end as F30
            , case 
                when
                    l9 in ('264000010','264000213','264000313','264000413','264000610')
                    then importh
                else 0
            end as F31
            , case 
                when
                    l4 in ('2620')
                    then importh
                else 0
            end as F32
            , case 
                when
                    l4 in ('2630')
                    then importh
                else 0
            end as F33
            , case 
                when
                    l6 in ('264002')
                    then importh
                else 0
            end as F34
            , case 
                when
                    l6 not in ('264021') and 
                    l7 not in ('2640201') and 
                    (
                        l5 in ('26402')
                    )
                    then importh
                else 0
            end as F40
            , case 
                when
                    l4 in ('2622')
                    then importh
                else 0
            end as F41
            , case 
                when
                    l4 in ('2632')
                    then importh
                else 0
            end as F42
            , case 
                when
                    l6 in ('264021')
                    then importh
                else 0
            end as F43
            , case 
                when
                    l5 not in ('46031','54041') and 
                    (
                        l4 in ('4603','5404','4690','5440','4611','5411','4610','5410')
                    )
                    then importh
                else 0
            end as D10
            , case 
                when
                    l4 in ('5470','5473') or
                    l5 in ('47420','47423')
                    then importh
                else 0
            end as D11
            , case 
                when
                    l4 in ('5482','5484') or
                    l5 in ('47432','47434')
                    then importh
                else 0
            end as D12
            , case 
                when
                    l5 in ('46031','54041')
                    then importh
                else 0
            end as D15
            , case 
                when
                    l5 not in ('46041','54051') and 
                    (
                        l4 in ('4604','5405')
                    )
                    then importh
                else 0
            end as D20
            , case 
                when
                    l4 in ('5474') or
                    l5 in ('47424')
                    then importh
                else 0
            end as D21
            , case 
                when
                    l4 in ('5485') or
                    l5 in ('47435')
                    then importh
                else 0
            end as D22
            , case 
                when
                    l5 in ('46041') or
                    l5 in ('54051')
                    then importh
                else 0
            end as D25
            , case 
                when
                    l5 not in ('46001','54001') and 
                    (
                        l4 in ('4600','5400')
                    )
                    then importh
                else 0
            end as D30
            , case 
                when
                    l4 in ('5471') or
                    l5 in ('47421')
                    then importh
                else 0
            end as D31
            , case 
                when
                    l4 in ('5481') or
                    l5 in ('47431')
                    then importh
                else 0
            end as D32
            , case 
                when
                    l5 in ('46001','54001')
                    then importh
                else 0
            end as D35
            , case 
                when
                    l5 not in ('46021','54021') and 
                    (
                        l4 in ('4602','5402')
                    )
                    then importh
                else 0
            end as D40
            , case 
                when
                    l4 in ('5472') or
                    l5 in ('47422')
                    then importh
                else 0
            end as D41
            , case 
                when
                    l4 in ('5483') or
                    l5 in ('47433')
                    then importh
                else 0
            end as D42
            , case 
                when
                    l5 in ('46021','54021')
                    then importh
                else 0
            end as D45
            , case 
                when
                    l5 not in ('46500','56900','46510','56910') and 
                    (
                        l3 in ('465','569')
                    )
                    then importh
                else 0
            end as D50
            , case 
                when
                    l5 in ('46500','56900','46510','56910')
                    then importh
                else 0
            end as D52
            , case 
                when
                    l5 not in ('54031') and 
                    (
                        l4 in ('5403')
                    )
                    then importh
                else 0
            end as AS0
            , case 
                when
                    l5 in ('54031')
                    then importh
                else 0
            end as AS1



       from
        (
            select A.*
                , substr(ctacgbal,1,2) as l2
                , substr(ctacgbal,1,3) as l3
                , substr(ctacgbal,1,4) as l4
                , substr(ctacgbal,1,5) as l5
                , substr(ctacgbal,1,6) as l6
                , substr(ctacgbal,1,7) as l7
                , substr(ctacgbal,1,8) as l8
                , substr(ctacgbal,1,9) as l9
            from {schema}.jm_posic_contr A
            where data_date_part='{fecha}'
        ) A
    ),

    saldos_contra as
    (
        select  feoperac
                ,  s1emp
                , contra1
                , sum(S10) as S10
                , sum(S11) as S11
                , sum(S12) as S12
                , sum(S13) as S13
                , sum(S14) as S14
                , sum(S15) as S15
                , sum(S16) as S16
                , sum(S10) as S20
                , sum(S11) as S21
                , sum(S12) as S22
                , sum(S13) as S23
                , sum(S14) as S24
                , sum(S15) as S25
                , sum(S16) as S26
                , sum(S30) as S30
                , sum(S31) as S31
                , sum(S32) as S32
                , sum(S33) as S33
                , sum(S34) as S34
                , sum(S35) as S35
                , sum(S36) as S36
                , sum(S40) as S40
                , sum(S43) as S43
                , sum(S51) as S51
                , sum(S52) as S52
                , sum(S53) as S53
                , sum(S54) as S54
                , sum(S55) as S55
                , sum(S56) as S56
                , sum(S57) as S57
                , sum(S61) as S61
                , sum(ARS) as ARS
                , sum(F10) as F10
                , sum(F11) as F11
                , sum(F12) as F12
                , sum(F13) as F13
                , sum(F14) as F14
                , sum(F20) as F20
                , sum(F21) as F21
                , sum(F22) as F22
                , sum(F23) as F23
                , sum(F24) as F24
                , sum(F30) as F30
                , sum(F31) as F31
                , sum(F32) as F32
                , sum(F33) as F33
                , sum(F34) as F34
                , sum(F40) as F40
                , sum(F41) as F41
                , sum(F42) as F42
                , sum(F43) as F43
                , sum(D10) as D10
                , sum(D11) as D11
                , sum(D12) as D12
                , sum(D15) as D15
                , sum(D20) as D20
                , sum(D21) as D21
                , sum(D22) as D22
                , sum(D25) as D25
                , sum(D30) as D30
                , sum(D31) as D31
                , sum(D32) as D32
                , sum(D35) as D35
                , sum(D40) as D40
                , sum(D41) as D41
                , sum(D42) as D42
                , sum(D45) as D45
                , sum(D50) as D50
                , sum(D52) as D52
                , sum(AS0) as AS0
                , sum(AS1) as AS1
        from saldos 
        group by 
                feoperac
                ,  s1emp
                , contra1
    ),


    stage_IFRS9 as 
    (
            select  feoperac, s1emp, contra1, stage, provisio
            from {schema}.jm_cto_ifrs9 
            where data_date_part='{fecha}' and tajifrs9 = 1

    ),

    catego as
    (
        select  a.feoperac
                , a.s1emp
                , a.contra1
                , a.id_prod
                , (sum(coalesce(on_balance_reg,0))) as on_balance_reg
                , (sum(coalesce(off_balance_reg,0))) as off_balance_reg
                , (sum(coalesce(on_balance_reg_old,0))) as on_balance_reg_old
                , (sum(coalesce(off_balance_reg_old,0))) as off_balance_reg_old
                , sum(coalesce(exp_orig,0)) as exp_orig
                , sum(coalesce(ead_reg,0)) as ead_reg
                , sum(coalesce(rwa_reg,0)) as rwa_reg
                , sum(coalesce(provdbal,0)) as provdbal
                , sum(coalesce(PROVFBAL,0)) as PROVFBAL
                , sum(coalesce(provisiones,0)) as provisiones
                ,approach
                ,b.stage
                , sum(coalesce(b.provisio,0)) as provisio

            from
            (
                select feoperac 
                    , s1emp 
                    , contra1
                    , id_prod
                    , sum(coalesce(decode(ind_mmff,'1',0,-isdebala),0)) as on_balance_reg_old
                    , sum(coalesce(decode(ind_mmff,'1',0,-isfubala),0)) as off_balance_reg_old
                    , sum(coalesce(decode(ind_mmff,'1',eadfinal, (abs(LEAST(isdebala,0)))),0)) as on_balance_reg
                    , sum(coalesce(decode(ind_mmff,'1',0, (abs(LEAST(isfubala,0)))),0)) as off_balance_reg
                    , sum(coalesce(exp_orig,0)) as exp_orig
                    , sum(coalesce(eadfinal,0)) as ead_reg
                    , sum(coalesce(criavesc,0))*12.5 as rwa_reg            
                    , sum(coalesce(provdbal,0)) as provdbal
                    , sum(coalesce(PROVFBAL,0)) as PROVFBAL
                    , sum(provdbal) + sum(PROVFBAL) as provisiones
                    ,"IRB" as approach

                from {schema}.jm_cto_aj_reg
                where data_date_part='{fecha}'

                    and 
                    ( 
                        (motexinf in (0) and tipoajus not in (1) and trim(cod_spv) = '') or
                        (motexinf in (20021,20023,19) and cat_msta in (8,17,28,30) and c_rw_irb = 0)
                    )
                group by feoperac
                    , s1emp
                    , contra1
                    , id_prod
                    , approach
                    ,ind_mmff

                union all
                select  
                      feoperac
                    , s1emp 
                    , contra1 
                    , id_prod
                    , sum(coalesce(decode(flg_mmff,1,0,-sdbpr),0)) as on_balance_reg_old
                    , sum(coalesce(decode(flg_mmff,1,0,-sfbpr),0)) as off_balance_reg_old
                    , sum(case 
                               when CATPRFIN in (37,47,48) then -SDBPR + PROVDBPR
                               when FLGEADNG in (0) then (abs(LEAST(SDBPR,0)))
                               else   -SDBPR end) as on_balance_reg
                    , sum(case 
                               when FLGEADNG in (0) then (abs(LEAST(SFBPR,0)))
                                   else   -SFBPR end) as oFF_balance_reg                
                    , sum(case 
                               when CATPRFIN in (37,47,48) and FLGEADNG in (0) then abs(LEAST(-SDBPR + PROVDBPR,0))+ abs(LEAST(-SDBPR + PROVDBPR,0))
                               when CATPRFIN in (37,47,48) and FLGEADNG in (1) then  (-SDBPR + PROVDBPR + -SFBPR)
                               when FLGEADNG in (0) then abs(LEAST(SDBPR,0))+ abs(LEAST(SFBPR,0))
                               else   -(SDBPR + SFBPR) end) as exp_orig             
                    , sum(coalesce(eadcffps,0)) as ead_reg
                    , sum(coalesce(rwa_posm,0)) as rwa_reg
                    , sum(case 
                               when CATPRFIN in (37,47,48) then 0
                               else  PROVDBPR end) as provdbal
                    ,sum(coalesce(PROVFBPR,0)) as PROVFBAL
                    ,sum(case  when CATPRFIN in (37,47,48) and FLGEADNG in (0) then least(abs(least(SDBPR + PROVDBPR,0)), 0) + least(abs(least(SFBPR,0)), PROVFBPR)
                               when CATPRFIN in (37,47,48) and FLGEADNG in (1) then PROVFBPR    
                               when FLGEADNG in (0) then least(abs(least(SDBPR,0)), PROVDBPR) + least(abs(least(SFBPR,0)), PROVFBPR)
                               when FLGEADNG in (1) then PROVDBPR + PROVFBPR
                               else 0 end) as provisiones
                    ,"STD" as approach
                from {schema}.jm_amp_aj_std
                where data_date_part='{fecha}'

                    and     
                    (
                       motexinf in (10020,20021,20021,0)
                    )
                group by feoperac
                    , s1emp
                    , contra1
                    , id_prod
                    , approach

            ) A
            LEFT JOIN (select distinct feoperac,contra1,s1emp,stage,provisio from stage_ifrs9) B
            on a.feoperac = b.feoperac and a.contra1=b.contra1 and a.s1emp=b.s1emp

            group by a.feoperac
                , a.s1emp
                , a.contra1
                , a.id_prod
                , approach
                , b.stage
    ),

    cto_raw_risk_metrics_new_inner as
    (
        select    a.feoperac
                , a.s1emp
                , a.contra1
                , stage
                , id_prod
                , '{divisa}' as coddiv
                , sum(coalesce(rwa_reg,0)) as rwa_reg
                , sum(on_balance_reg) as on_balance_reg
                , sum(off_balance_reg) as off_balance_reg
                , sum(on_balance_reg_old) as on_balance_reg_old
                , sum(off_balance_reg_old) as off_balance_reg_old
                , sum(exp_orig) as exp_orig
                , sum(ead_reg) as ead_reg
                , sum(provdbal) as provdbal
                , sum(PROVFBAL) as PROVFBAL
                , sum(provisiones) as provisiones
                , sum(coalesce(provisio,0)) as provisio
                , sum(decode(stage, 2, 0, S10)) as S10
                , sum(decode(stage, 2, 0, S11)) as S11
                , sum(decode(stage, 2, 0, S12)) as S12
                , sum(decode(stage, 2, 0, S13)) as S13
                , sum(decode(stage, 2, 0, S14)) as S14
                , sum(decode(stage, 2, 0, S15)) as S15
                , sum(decode(stage, 2, 0, S16)) as S16
                , sum(decode(stage, 2, S10, 0)) as S20
                , sum(decode(stage, 2, S11, 0)) as S21
                , sum(decode(stage, 2, S12, 0)) as S22
                , sum(decode(stage, 2, S13, 0)) as S23
                , sum(decode(stage, 2, S14, 0)) as S24
                , sum(decode(stage, 2, S15, 0)) as S25
                , sum(decode(stage, 2, S16, 0)) as S26
                , sum(S30) as S30
                , sum(S31) as S31
                , sum(S32) as S32
                , sum(S33) as S33
                , sum(S34) as S34
                , sum(S35) as S35
                , sum(S36) as S36
                , sum(S40) as S40
                , sum(S43) as S43
                , sum(S51) as S51
                , sum(S52) as S52
                , sum(S53) as S53
                , sum(S54) as S54
                , sum(S55) as S55
                , sum(S56) as S56
                , sum(S57) as S57
                , sum(S61) as S61
                , sum(ARS) as ARS
                , sum(F10) as F10
                , sum(F11) as F11
                , sum(F12) as F12
                , sum(F13) as F13
                , sum(F14) as F14
                , sum(F20) as F20
                , sum(F21) as F21
                , sum(F22) as F22
                , sum(F23) as F23
                , sum(F24) as F24
                , sum(F30) as F30
                , sum(F31) as F31
                , sum(F32) as F32
                , sum(F33) as F33
                , sum(F34) as F34
                , sum(F40) as F40
                , sum(F41) as F41
                , sum(F42) as F42
                , sum(F43) as F43
                , sum(D10) as D10
                , sum(D11) as D11
                , sum(D12) as D12
                , sum(D15) as D15
                , sum(D20) as D20
                , sum(D21) as D21
                , sum(D22) as D22
                , sum(D25) as D25
                , sum(D30) as D30
                , sum(D31) as D31
                , sum(D32) as D32
                , sum(D35) as D35
                , sum(D40) as D40
                , sum(D41) as D41
                , sum(D42) as D42
                , sum(D45) as D45
                , sum(D50) as D50
                , sum(D52) as D52
                , sum(AS0) as AS0
                , sum(AS1) as AS1
                , cast(from_unixtime(unix_timestamp(), 'yyyyMMddHHmmss') as bigint) as process_timestamp

        from saldos_contra A
        inner join catego B on
                        A.feoperac = B.feoperac
                    and A.s1emp=B.s1emp
                    and A.contra1=B.contra1
        group by 
                  A.feoperac
                , A.s1emp
                , A.contra1
                , B.ID_PROD
                , B.stage
                , coddiv
    ),

    cto_risk_metrics as
    (
        select  a.*,b.provifrs9, c.tpcam

            , S10 + S12 
            + S20 + S22 
            + S30 + S31 + S32 + S33
            + S40
            + S51 + S54 + S57
            + S61
            as Riesgo_dispuesto
            , S30 + S31 + S32 + S33
            + S40
            + S54 + S57
            as Saldo_dudoso_S3_noper
            , S10 + S11 + S12 
            + S20 + S21 + S22 
            + S30 + S31 + S32 + S33
            + S40
            + S51 + S54 + S57
            + S61
            as Exposicion_total_denom_tasa_mora

        from cto_raw_risk_metrics_new_inner A
            left join 
                (
                    select data_date_part as feoperac, s1emp, contra1, sum(importh) as provifrs9
                    from {schema}.jm_prov_esotr 
                    where data_date_part='{fecha}'
                        and tip_impt=2
                    group by data_date_part, s1emp, contra1
                ) B
         on
                A.feoperac = B.feoperac
            and A.s1emp=B.s1emp
            and A.contra1=B.contra1
            left join 
                (
                    select data_date_part as feoperac,coddiv,TPCAM
                    from {schema}.jm_tip_cambio 
                    where data_date_part='{fecha}' and coddiv2 = 'EUR'
                ) C
         on
                A.feoperac = B.feoperac
                and A.coddiv=C.coddiv
    ),  

    adn_group as (
    select a.feoperac, a.s1emp, a.contra1, sum(a.porcen) as total_porcen_cont
    from {schema}.jm_cto_adnnpp a
    where a.feoperac = '{fecha}'
    group by 1,2,3
    ),

    final_adn as (
    select distinct a.feoperac, a.s1emp, a.contra1, trim(a.adn) as adn, 
    case 
    when trim(a.adn) = 'AN' then 'Total area de Negocio' 
    when trim(a.adn) = 'WM' then 'Wealth Management and Insurance' 
    when trim(a.adn) = 'BP' then 'Banca Privada' 
    when trim(a.adn) = 'AN01010100' then 'Private Wealth' 
    when trim(a.adn) = 'AN01050000' then 'Resto Banca Privada' 
    when trim(a.adn) = 'GA' then 'Gestion de Activos' 
    when trim(a.adn) = 'AN03010000' then 'Fondos de Inversion' 
    when trim(a.adn) = 'AN03020000' then 'Fondos de Pensiones' 
    when trim(a.adn) = 'SE' then 'Seguros' 
    when trim(a.adn) = 'AN03030000' then 'Seguros.' 
    when trim(a.adn) = 'BC00' then 'Particulares' 
    when trim(a.adn) = 'AN01010201' then 'Select' 
    when trim(a.adn) = 'AN01010500' then 'Resto Particulares' 
    when trim(a.adn) = 'BC10' then 'Pymes, Empresas e Instituciones' 
    when trim(a.adn) = 'BC1010' then 'Empresas' 
    when trim(a.adn) = 'AN01020210' then 'Empresas.' 
    when trim(a.adn) = 'BC1014' then 'Pymes' 
    when trim(a.adn) = 'AN01020200' then 'Pymes II' 
    when trim(a.adn) = 'BC1015' then 'Pymes I' 
    when trim(a.adn) = 'AN01020410' then 'Pymes I.' 
    when trim(a.adn) = 'BC1020' then 'Instituciones' 
    when trim(a.adn) = 'AN01040220' then 'Instituciones.' 
    when trim(a.adn) = 'FC' then 'Segmento Financiacion al Consumo' 
    when trim(a.adn) = 'AN06010000' then 'Consumo Auto' 
    when trim(a.adn) = 'AN06020000' then 'Consumo No Auto' 
    when trim(a.adn) = 'GT' then 'PagoNxt' 
    when trim(a.adn) = 'AN08010000' then 'PagoNxt.' 
    when trim(a.adn) = 'AN08020000' then 'PagoNxt Consumer' 
    when trim(a.adn) = 'AN08030000' then 'PagoNxt Trade' 
    when trim(a.adn) = 'AN08040000' then 'PagoNxt Merchant' 
    when trim(a.adn) = 'GB' then 'Corporate and Investment Banking (CIB)' 
    when trim(a.adn) = 'GB05A' then 'Corporate Finance' 
    when trim(a.adn) = 'GB0505' then 'M & A - Mergers and Acquisitions' 
    when trim(a.adn) = 'AN02050600' then 'M & A - Mergers and Acquisitions.' 
    when trim(a.adn) = 'GB0510' then 'ECM - Equity Capital Markets' 
    when trim(a.adn) = 'AN02050500' then 'ECM - Equity Capital Markets.' 
    when trim(a.adn) = 'GB0515' then 'Corporate Equity Derivatives' 
    when trim(a.adn) = 'AN02050800' then 'Corporate Equity Derivatives.' 
    when trim(a.adn) = 'GB0520' then 'Emisores' 
    when trim(a.adn) = 'AN02050700' then 'Emisores.' 
    when trim(a.adn) = 'GB05' then 'Global Debt Financing' 
    when trim(a.adn) = 'GB0525' then 'Project and Acquisition Finance' 
    when trim(a.adn) = 'AN02030300' then 'Project and Acquisition Finance.' 
    when trim(a.adn) = 'GB0540' then 'Sindicados' 
    when trim(a.adn) = 'AN02030400' then 'Sindicados.' 
    when trim(a.adn) = 'GB0530' then 'Debt Capital Markets & Securitization' 
    when trim(a.adn) = 'AN02030200' then 'Debt Capital Markets & Securitization.' 
    when trim(a.adn) = 'GB0535' then 'ACS - Asset & Capital Structuring' 
    when trim(a.adn) = 'AN02010200' then 'ACS - Asset & Capital Structuring.' 
    when trim(a.adn) = 'GB0545' then 'Leveraged Finance' 
    when trim(a.adn) = 'AN02030600' then 'Leveraged Finance.' 
    when trim(a.adn) = 'GB0550' then 'Other Global Debt Financing' 
    when trim(a.adn) = 'AN02030700' then 'Other Global Debt Financing.' 
    when trim(a.adn) = 'GB10' then 'GTB - Global Transaction Banking' 
    when trim(a.adn) = 'GB1005' then 'Cash Management' 
    when trim(a.adn) = 'AN02020100' then 'Cash Management.' 
    when trim(a.adn) = 'GB1010' then 'Financiacion Basica' 
    when trim(a.adn) = 'AN02020300' then 'Financiacion Basica.' 
    when trim(a.adn) = 'GB1015' then 'Trade & WCS' 
    when trim(a.adn) = 'AN02020200' then 'Trade & WCS.' 
    when trim(a.adn) = 'GB1025' then 'Export & Agency Finance' 
    when trim(a.adn) = 'AN02020400' then 'Export & Agency Finance.' 
    when trim(a.adn) = 'GB1020' then 'Custodia' 
    when trim(a.adn) = 'AN02050300' then 'Custodia.' 
    when trim(a.adn) = 'GB20' then 'Global Markets' 
    when trim(a.adn) = 'GB2005' then 'Sales Mercados' 
    when trim(a.adn) = 'AN02080000' then 'Sales Mercados.' 
    when trim(a.adn) = 'AN02090000' then 'Market Making' 
    when trim(a.adn) = 'GB2010' then 'ETD - Exchange Traded Derivatives' 
    when trim(a.adn) = 'AN02050100' then 'ETD - Exchange Traded Derivatives.' 
    when trim(a.adn) = 'GB2015' then 'Cash Equity' 
    when trim(a.adn) = 'AN02050400' then 'Cash Equity.' 
    when trim(a.adn) = 'AN02030900' then 'XVA' 
    when trim(a.adn) = 'AN02030500' then 'Global Structuring' 
    when trim(a.adn) = 'AN02060000' then 'Other Market Products' 
    when trim(a.adn) = 'GB45' then 'ACPM' 
    when trim(a.adn) = 'AN02030045' then 'ACPM.' 
    when trim(a.adn) = 'NC' then 'SIN USO Non-core' 
    when trim(a.adn) = 'AC' then 'Actividades Corporativas' 
    when trim(a.adn) = 'AC05' then 'Gestion Financiera' 
    when trim(a.adn) = 'AN04010100' then 'ALCO' 
    when trim(a.adn) = 'AN04010400' then 'Coberturas de balance' 
    when trim(a.adn) = 'AN04010500' then 'Financiacion' 
    when trim(a.adn) = 'AN04010600' then 'Coberturas estrategicas' 
    when trim(a.adn) = 'AN04010700' then 'Financiacion intragrupo' 
    when trim(a.adn) = 'AC10' then 'Resto' 
    when trim(a.adn) = 'AN04020000' then 'Pool de Fondos' 
    when trim(a.adn) = 'AC1010' then 'Otros' 
    when trim(a.adn) = 'AN04030000' then 'Resto Actividades Corporativas' 
    when trim(a.adn) = 'AN04040000' then 'area de Reestructuracion' 
    when trim(a.adn) = 'AN04050000' then 'area de Reestructuracion Anular' 
    when trim(a.adn) = 'EL' then 'Reasignacion del Intragrupo' 
    else 'No Informado'
    end as adn_desc,
    a.porcen, b.total_porcen_cont,
    round((a.porcen / nullifzero(b.total_porcen_cont)) * 100,3) as porcen_adn
    from {schema}.jm_cto_adnnpp a
    left join adn_group b on a.feoperac = b.feoperac and a.s1emp = b.s1emp and a.contra1 = b.contra1 and a.feoperac = '{fecha}'
    where a.feoperac = '{fecha}'
    ),

    fin_adn as
    (
    select 
           adn,
           case when adn = 'AN' then ''
    when adn = 'WM' then 'WM'
    when adn = 'BC00' then 'Rest Individuals'
    when adn = 'BC10' then 'SME Corporates & Institutions'
    when adn = 'FC' then 'Consumo Auto'
    when adn = 'GT' then 'Pago Nxt'
    when adn = 'GB' then 'SCIB'
    when adn = 'AC' then 'Others'
    when adn = 'EL' then 'Others'
    when adn = 'GB10' then 'SCIB'
    when adn = 'GB20' then 'SCIB'
    when adn = 'BP' then 'WM'
    when adn = 'GA' then 'WM'
    when adn = 'SE' then 'WM'
    when adn = 'AN01010201' then 'Rest Individuals'
    when adn = 'AN01010500' then 'Rest Individuals'
    when adn = 'BC1010' then 'Corporates'
    when adn = 'BC1014' then 'SME'
    when adn = 'BC1020' then 'Institutions'
    when adn = 'AN06010000' then 'Consumo Auto'
    when adn = 'AN06020000' then 'Consumo No Auto'
    when adn = 'AN06030000' then 'Consumo Auto'
    when adn = 'AN08010000' then 'Pago Nxt'
    when adn = 'AN08020000' then 'Pago Nxt'
    when adn = 'AN08030000' then 'Pago Nxt'
    when adn = 'AN08040000' then 'Pago Nxt'
    when adn = 'GB05A' then 'SCIB'
    when adn = 'GB05' then 'SCIB'
    when adn = 'AN02030500' then 'SCIB'
    when adn = 'GB1020' then 'SCIB'
    when adn = 'GB45' then 'SCIB'
    when adn = 'AC05' then 'Others'
    when adn = 'AC10' then 'Others'
    when adn = 'GB0545' then 'SCIB'
    when adn = 'GB0550' then 'SCIB'
    when adn = 'GB1005' then 'SCIB'
    when adn = 'GB1010' then 'SCIB'
    when adn = 'GB1015' then 'SCIB'
    when adn = 'GB1025' then 'SCIB'
    when adn = 'GB2005' then 'SCIB'
    when adn = 'AN02090000' then 'SCIB'
    when adn = 'GB2015' then 'SCIB'
    when adn = 'AN02030900' then 'SCIB'
    when adn = 'AN01010100' then 'WM'
    when adn = 'AN03020000' then 'WM'
    when adn = 'AN01020200' then 'SME 2'
    when adn = 'AN02050300' then 'SCIB'
    when adn = 'AN04010100' then 'Others'
    when adn = 'AN04020000' then 'Others'
    when adn = 'AN02020200' then 'SCIB'
    when adn = 'AN02080000' then 'SCIB'
    when adn = 'AN02050400' then 'SCIB'
    when adn = 'AN01020410' then 'SME 1'
    when adn = 'AN02050700' then 'SCIB'
    when adn = 'AN02030400' then 'SCIB'
    when adn = 'AN04030000' then 'Others'
    when adn = 'GB2010' then 'SCIB'
    when adn = 'AN02060000' then 'SCIB'
    when adn = 'AN01050000' then 'WM'
    when adn = 'AN03010000' then 'WM'
    when adn = 'AN03030000' then 'WM'
    when adn = 'AN01020210' then 'Corporates'
    when adn = 'BC1015' then 'SME 1'
    when adn = 'AN01040220' then 'Institutions'
    when adn = 'GB0505' then 'SCIB'
    when adn = 'GB0510' then 'SCIB'
    when adn = 'GB0515' then 'SCIB'
    when adn = 'GB0520' then 'SCIB'
    when adn = 'GB0525' then 'SCIB'
    when adn = 'GB0540' then 'SCIB'
    when adn = 'GB0530' then 'SCIB'
    when adn = 'GB0535' then 'SCIB'
    when adn = 'AN02030045' then 'SCIB'
    when adn = 'AN04010400' then 'Others'
    when adn = 'AN04010500' then 'Others'
    when adn = 'AN04010600' then 'Others'
    when adn = 'AN04010700' then 'Others'
    when adn = 'AC1010' then 'Others'
    when adn = 'AN02030600' then 'SCIB'
    when adn = 'AN02030700' then 'SCIB'
    when adn = 'AN02020100' then 'SCIB'
    when adn = 'AN02020300' then 'SCIB'
    when adn = 'AN02020400' then 'SCIB'
    when adn = 'AN02050100' then 'SCIB'
    when adn = 'AN02050600' then 'SCIB'
    when adn = 'AN02050500' then 'SCIB'
    when adn = 'AN02050800' then 'SCIB'
    when adn = 'AN02030300' then 'SCIB'
    when adn = 'AN02030200' then 'SCIB'
    when adn = 'AN02010200' then 'SCIB'
    when adn = 'CA' then 'Cards'
    when adn = 'AN07020000' then 'Cards'
    when adn = 'AN07030000' then 'Cards'
    else 'No informado'
    end as segmento_adn,
         a.*
    from cto_risk_metrics a
    left join final_adn d on a.s1emp = d.s1emp and a.contra1 = d.contra1
    ),

    tabla_fin as
    (
    select *
        , (-Riesgo_dispuesto *tpcam)/1000000 as Riesgo_dispuesto_eur
        , (on_balance_reg *tpcam)/1000000 as on_balance_reg_eur
        , (off_balance_reg *tpcam)/1000000 as off_balance_reg_eur
        , (exp_orig *tpcam)/1000000 as exp_orig_eur
        , (ead_reg *tpcam)/1000000 as ead_reg_eur
        , (Exposicion_total_denom_tasa_mora *tpcam)/1000000 as Exposicion_total_denom_tasa_mora_eur
        , (provdbal *tpcam)/1000000 as provdbal_eur
        , (PROVFBAl *tpcam)/1000000 as PROVFBAl_eur
        , (provisiones *tpcam)/1000000 as provisiones_eur
        , (provifrs9 *tpcam)/1000000 as provifrs9_eur
        , (provisio *tpcam)/1000000 as provisio_eur

        from fin_adn
    ),
    
    tabla_vf as
    (
    select 
           feoperac,
           stage,
           s1emp,
           contra1,
           segmento_adn,
           Riesgo_dispuesto_eur as dispuesto_cargabal,
        -- (-Saldo_dudoso_S3_noper)/1000000*0.533615 as saldos3,    
           on_balance_reg_eur as ob,
           case 
                    when stage in (3) then off_balance_reg_eur
                    else 0 
                                                                end as disponible_mora,
           case 
                    when ID_PROD in (3,4) then off_balance_reg_eur
                    else 0 
                                                                end as avales
       --  (calculated ob + calculated disponible_mora + calculated avales) as total_risk_cap,                                                    
          ,off_balance_reg_eur as off
          ,exp_orig_eur as exp_orig
          ,ead_reg_eur as ead_reg
          ,-Exposicion_total_denom_tasa_mora_eur as Exposicion_total_denom_tasa_mora
          ,provdbal_eur as provdbal
          ,PROVFBAl_eur as PROVFBAL
          ,provisiones_eur as provisiones
          ,provifrs9_eur as provifrs9
          ,provisio_eur as provisio
    --    ,Fondos_cobertura/1000000*0.0533615 as Fondos_cobertura
    --    ,Dotaciones/1000000*0.0533615 as Dotaciones

    from tabla_fin 
    ),
    
    tabla_metricas as 
    (
    select A.*,B.eadifrs9, B.provisio as prov, B.P12IFRS9, B.LGDIFRS9, B.PLTIFRS9
    from tabla_vf as A
    left join {schema}.jm_cto_ifrs9 as B 
    on A.s1emp = B.s1emp and A.contra1 = B.contra1 and A.feoperac = B.data_date_part
    )    
    select * from tabla_metricas
    """
    try:
        consulta = f"""drop table if exists default.perimetro_{pais}_{aaaamm}"""
        cnxn.cursor().execute(consulta)
        cnxn.cursor().execute(perimetro)
        df_temporal = pd.read_sql(f"""select * from default.perimetro_{pais}_{aaaamm}""", cnxn)
        return df_temporal
    except Exception as e:
        print(f'Error al ejecutar la consulta para la fecha {fecha}: {e}')
        raise

#####################
## EXTRACCION PERIMETRO ##
#####################
"""Variables"""
schemas = {
    'mex': 'cd_bdr_mex'
}

divisa="MXN"
cnxn = python_supra_conection("pro", "impala")

# Genera todas las fechas y las ordena
fechas = generar_fechas_ultimo_dia_mes(2023,2024)
fechas.sort()  # Asegura que las fechas estén en orden cronológico

# Usa una cola (deque) para mantener el orden de las fechas
fechas_pendientes = deque(fechas)
fechas_exitosas = set()
fechas_fallidas = deque()  # Usa una cola también para las fechas fallidas para mantener el orden de reintento

# Ejecución para una fecha
#fecha_actual = fechas_pendientes.popleft()
df_final = pd.DataFrame()
perimetro = pd.DataFrame()

for pais, schema in schemas.items():
    print(f'Ejecutando para: {pais.upper()}')
    while fechas_pendientes or fechas_fallidas:
        if fechas_pendientes:
            fecha_actual = fechas_pendientes.popleft()  # Obtiene y elimina la primera fecha de la cola
        else:
            fecha_actual = fechas_fallidas.popleft()  # Si no hay pendientes, sigue con las fallidas
        try:
            aaaamm = fecha_actual[:7].replace('-','')
            df_temporal = ejecutar_consulta_sql(fecha_actual, schema, divisa, pais, aaaamm, cnxn)
            if not df_temporal.empty:
                fechas_exitosas.add(fecha_actual)
                print(f'Consulta exitosa para la fecha: {fecha_actual}')
            else:
                print(f'Consulta ejecutada, pero sin datos para la fecha: {fecha_actual}')
                fechas_fallidas.append(fecha_actual)  # Añade la fecha al final de la cola de fallidas
        except Exception as e:
            print(f'Error en la consulta de la fecha {fecha_actual}: {e}')
            fechas_fallidas.append(fecha_actual)  # Añade la fecha al final de la cola de fallidas