CREATE TABLE default.perimetro_{pais}_{aaaamm} AS
    WITH cte_saldos AS (
        SELECT
        A.*,
        CASE
        WHEN l4 NOT IN ('1840') AND l5 NOT IN ('12201','17320','17323') AND l6 NOT IN ('122113','122213','122313','124010','124113','124213','124313','173210','173220','173240','173252','173290') AND (l3 IN ('122','124','184') OR l4 IN ('1732') OR l5 IN ('17981')) THEN importh
    ELSE 0
    END AS S10,
    CASE
    WHEN l3 NOT IN ('304') AND l4 NOT IN ('3022','3021') AND l5 NOT IN ('30002','30012','30022','30032','30302','30392') AND l2 IN ('30') THEN importh
    ELSE 0
    END AS S12,
    CASE
    WHEN l4 NOT IN ('1401') AND l5 NOT IN ('14022','14032','14042','14051','14500') AND l6 NOT IN ('140010','140213','140313','140413','143010','143213','143313','143413') AND l7 NOT IN ('1405010','1405012','1435010') AND (l3 IN ('140') OR l3 IN ('143') OR l3 IN ('145')) THEN importh
    ELSE 0
    END AS S30,
    CASE
    WHEN l4 IN ('1461','1463') OR l6 IN ('140010','140213','140313','140413','143010','143213','143313','143413') OR l7 IN ('1405010','1435010') THEN importh
    ELSE 0
    END AS S31,
    CASE
    WHEN l5 NOT IN ('30402','30412','30422','30435','30436','30442','30446') AND l3 IN ('304') THEN importh
    ELSE 0
    END AS S32,
    CASE
    WHEN l5 NOT IN ('31702','31706','31712','31716') AND l3 IN ('317') THEN importh
    ELSE 0
    END AS S33,
    CASE
    WHEN l5 IN ('14022','14032','14042') OR l7 IN ('1405012') THEN importh
    ELSE 0
    END AS S40,
    CASE
    WHEN l6 NOT IN ('173101','173110') AND (l3 IN ('121','123') OR l4 IN ('1731','1031','1033') OR l7 IN ('1797201','1797221')) THEN importh
    ELSE 0
    END AS S51,
    CASE
    WHEN l4 NOT IN ('1421','1441') AND l7 NOT IN ('1425012') AND (l3 IN ('142') OR l3 IN ('144') OR l4 IN ('1460') OR l4 IN ('1462')) THEN importh
    ELSE 0
    END AS S54,
    CASE
    WHEN l7 IN ('1425012') THEN importh
    ELSE 0
    END AS S57,
    CASE
    WHEN l6 NOT IN ('173001','173010') AND (l3 IN ('120') OR l4 IN ('1730','1030')) THEN importh
    ELSE 0
    END AS S61
    FROM (
             SELECT
             A.*,
             SUBSTR(ctacgbal,1,2) AS l2,
    SUBSTR(ctacgbal,1,3) AS l3,
    SUBSTR(ctacgbal,1,4) AS l4,
    SUBSTR(ctacgbal,1,5) AS l5,
    SUBSTR(ctacgbal,1,6) AS l6,
    SUBSTR(ctacgbal,1,7) AS l7,
    SUBSTR(ctacgbal,1,8) AS l8,
    SUBSTR(ctacgbal,1,9) AS l9
    FROM
    cd_bdr_{BDR}.jm_posic_contr A
    WHERE
    data_date_part = '{fecha}'
    ) A
    ),
    cte_stage_IFRS9 AS (
                           SELECT
                           feoperac,
                           s1emp,
                           contra1,
                           stage,
                           CASE WHEN provisio < -99999999 THEN 0 ELSE provisio END AS provisio
                           FROM
                           cd_bdr_{BDR}.jm_cto_ifrs9
                           WHERE
                           data_date_part = '{fecha}' AND
                           tajifrs9 = 1
                                  ),
    cte_catego AS (
                      SELECT
                      A.feoperac,
                      A.s1emp,
                      A.contra1,
                      A.id_prod,
                      A.idnumcli,
                      SUM(COALESCE(on_balance_reg,0)) AS on_balance_reg,
    SUM(COALESCE(off_balance_reg,0)) AS off_balance_reg,
    SUM(COALESCE(exp_orig,0)) AS exp_orig,
    SUM(COALESCE(ead_reg,0)) AS ead_reg,
    SUM(COALESCE(rwa_reg,0)) AS rwa_reg,
    SUM(COALESCE(provdbal,0)) AS provdbal,
    SUM(COALESCE(PROVFBAL,0)) AS PROVFBAL,
    SUM(COALESCE(provisiones,0)) AS provisiones,
    SUM(COALESCE(b.provisio,0)) AS provisio,
    cli_type_id,
    SECURED
    FROM (
             SELECT
             A.feoperac,
             A.s1emp,
             A.contra1,
             A.idnumcli,
             id_prod,
             SUM(COALESCE(DECODE(ind_mmff,'1',eadfinal,ABS(LEAST(isdebala,0))),0)) AS on_balance_reg,
    SUM(COALESCE(DECODE(ind_mmff,'1',0,ABS(LEAST(isfubala,0))),0)) AS off_balance_reg,
    SUM(COALESCE(exp_orig,0)) AS exp_orig,
    SUM(COALESCE(eadfinal,0)) AS ead_reg,
    SUM(COALESCE(criavesc,0))*12.5 AS rwa_reg,
    SUM(COALESCE(provdbal,0)) AS provdbal,
    SUM(COALESCE(PROVFBAL,0)) AS PROVFBAL,
    SUM(provdbal) + SUM(PROVFBAL) AS provisiones,
    tip_pert AS cli_type_id,
    CASE WHEN NVL(ind_secured_re_ctr,0) = 1 THEN 1 ELSE 0 END AS SECURED
    FROM
    cd_bdr_{BDR}.jm_cto_aj_reg A
    LEFT JOIN (
                  SELECT
                  s1emp,
                  contra1,
                  CASE WHEN SUM(NVL(val_gar.imgarant,0)) > 0 THEN 1 ELSE 0 END AS ind_secured_re_ctr
    FROM
    cd_bdr_{BDR}.jm_garant_cto ctr_gar
    LEFT JOIN
    cd_bdr_{BDR}.jm_val_gara val_gar ON val_gar.feoperac = ctr_gar.feoperac
    AND val_gar.s1emp = ctr_gar.s1emp
    AND val_gar.biengar1 = ctr_gar.biengar1
    WHERE
    ctr_gar.tip_gara = 2
    AND ctr_gar.cod_gar IN (11,25,81,82,83,84,85,86,87,88,89,90)
    AND ctr_gar.feoperac = '{fecha}'
    GROUP BY
    ctr_gar.feoperac,
    ctr_gar.s1emp,
    ctr_gar.contra1
    ) gar ON gar.s1emp = A.s1emp
    AND gar.feoperac = '{fecha}'
    AND gar.contra1 = A.contra1
    WHERE
    data_date_part = '{fecha}'
    AND (
(motexinf = 0 OR motexinf IS NULL)
    OR (motexinf IN (20021,20023,19) AND cat_msta IN (8,17,28,30) AND c_rw_irb = 0)
    )
    GROUP BY
    feoperac,
    s1emp,
    contra1,
    id_prod,
    idnumcli,
    cli_type_id,
    SECURED
    UNION ALL
    SELECT
    feoperac,
    a.s1emp,
    a.contra1,
    a.idnumcli,
    id_prod,
    SUM(
           CASE
           WHEN CATPRFIN IN (37,47,48) THEN -SDBPR + PROVDBPR
    WHEN FLGEADNG = 0 THEN ABS(LEAST(SDBPR,0))
    ELSE -SDBPR
    END
    ) AS on_balance_reg,
    SUM(
           CASE
           WHEN FLGEADNG = 0 THEN ABS(LEAST(SFBPR,0))
    ELSE -SFBPR
    END
    ) AS off_balance_reg,
    SUM(
           CASE
           WHEN CATPRFIN IN (37,47,48) AND FLGEADNG = 0 THEN ABS(LEAST(-SDBPR + PROVDBPR,0))+ ABS(LEAST(-SDBPR + PROVDBPR,0))
    WHEN CATPRFIN IN (37,47,48) AND FLGEADNG = 1 THEN -SDBPR + PROVDBPR + -SFBPR
    WHEN FLGEADNG = 0 THEN ABS(LEAST(SDBPR,0)) + ABS(LEAST(SFBPR,0))
    ELSE -(SDBPR + SFBPR)
    END
    ) AS exp_orig,
    SUM(COALESCE(eadcffps,0)) AS ead_reg,
    SUM(COALESCE(rwa_posm,0)) AS rwa_reg,
    SUM(
           CASE
           WHEN CATPRFIN IN (37,47,48) THEN 0
    ELSE PROVDBPR
    END
    ) AS provdbal,
    SUM(COALESCE(PROVFBPR,0)) AS PROVFBAL,
    SUM(
           CASE
           WHEN CATPRFIN IN (37,47,48) AND FLGEADNG = 0 THEN LEAST(ABS(LEAST(SDBPR + PROVDBPR,0)),0) + LEAST(ABS(LEAST(SFBPR,0)),PROVFBPR)
    WHEN CATPRFIN IN (37,47,48) AND FLGEADNG = 1 THEN PROVFBPR
    WHEN FLGEADNG = 0 THEN LEAST(ABS(LEAST(SDBPR,0)),PROVDBPR) + LEAST(ABS(LEAST(SFBPR,0)),PROVFBPR)
    WHEN FLGEADNG = 1 THEN PROVDBPR + PROVFBPR
    ELSE 0
    END
    ) AS provisiones,
    tipersti AS cli_type_id,
    COALESCE(secured,0) AS secured
    FROM
    cd_bdr_{BDR}.jm_amp_aj_std A
    LEFT JOIN (
                  SELECT
                  s1emp,
                  contra1,
                  CASE WHEN sbcprfin IN (4,5) THEN 1 ELSE 0 END AS SECURED
    FROM
    cd_bdr_{BDR}.jm_amp_aj_std
    WHERE
    data_date_part = '{fecha}'
    AND sbcprfin IN (4,5)
    AND motexinf != 50065
    ) B ON A.s1emp = B.s1emp
    AND A.contra1 = B.contra1
    WHERE
    data_date_part = '{fecha}'
    AND motexinf != 50065
    GROUP BY
    feoperac,
    a.s1emp,
    a.contra1,
    id_prod,
    cli_type_id,
    a.idnumcli,
    secured
    ),
    cte_catego2 AS (
                       SELECT
                       A.feoperac,
                       A.s1emp,
                       A.contra1,
                       a.idnumcli,
                       id_prod,
                       SUM(on_balance_reg) AS on_balance_reg,
    SUM(off_balance_reg) AS off_balance_reg,
    SUM(exp_orig) AS exp_orig,
    SUM(ead_reg) AS ead_reg,
    SUM(provdbal) AS provdbal,
    SUM(PROVFBAL) AS PROVFBAL,
    SUM(provisiones) AS provisiones,
    cli_type_id,
    b.secured,
    clisegm
    FROM
    cte_catego A
    LEFT JOIN (
                  SELECT
                  contra1,
                  s1emp,
                  SECURED
                  FROM
                  cte_catego
                  WHERE
                  secured IN (1)
    ) B ON A.s1emp = B.s1emp
    AND A.contra1 = B.contra1
    LEFT JOIN (
                  SELECT
                  feoperac,
                  s1emp,
                  idnumcli,
                  clisegm
                  FROM
                  cd_bdr_{BDR}.jm_client_bii
                  WHERE
                  data_date_part = '{fecha}'
                         ) C ON A.s1emp = C.s1emp
    AND A.idnumcli = C.idnumcli
    AND A.feoperac = C.feoperac
    GROUP BY
    A.feoperac,
    A.s1emp,
    A.contra1,
    id_prod,
    cli_type_id,
    a.idnumcli,
    b.secured,
    clisegm
    ),
    cte_cto_raw_risk_metrics_new_inner AS (
                                              SELECT
                                              a.feoperac,
                                              a.s1emp,
                                              a.contra1,
                                              b.idnumcli,
                                              stage,
                                              id_prod
    ,cli_type_id
    ,secured
    ,clisegm
    ,'{divisa}' AS coddiv
    ,CASE WHEN b.contra1 IS NULL THEN 0 ELSE 1 END AS perimetro_catego
    ,SUM(on_balance_reg) AS on_balance_reg
    ,SUM(off_balance_reg) AS off_balance_reg
    ,SUM(exp_orig) AS exp_orig
    ,SUM(ead_reg) AS ead_reg
    ,SUM(provdbal) AS provdbal
    ,SUM(PROVFBAL) AS PROVFBAL
    ,SUM(provisiones) AS provisiones
    ,SUM(S10) AS S10
    ,SUM(S12) AS S12
    ,SUM(S30) AS S30
    ,SUM(S31) AS S31
    ,SUM(S32) AS S32
    ,SUM(S33) AS S33
    ,SUM(S40) AS S40
    ,SUM(S51) AS S51
    ,SUM(S54) AS S54
    ,SUM(S57) AS S57
    ,SUM(S61) AS S61
    ,CAST(FROM_UNIXTIME(UNIX_TIMESTAMP(),'yyyyMMddHHmmss') AS BIGINT) AS process_timestamp
    FROM
    saldos_contra A
    FULL JOIN cte_catego2 B ON A.feoperac = B.feoperac
    AND A.s1emp = B.s1emp
    AND A.contra1 = B.contra1
    LEFT JOIN (
                  SELECT DISTINCT
                  feoperac,
                  contra1,
                  s1emp,
                  stage,
                  provisio
                  FROM
                  stage_ifrs9
              ) C ON A.feoperac = C.feoperac
    AND A.contra1 = C.contra1
    AND A.s1emp = C.s1emp
    GROUP BY
    A.feoperac,
    A.s1emp,
    A.contra1,
    perimetro_catego,
    B.id_prod,
    C.stage,
    coddiv,
    secured,
    cli_type_id,
    b.idnumcli,
    clisegm
    ),
    cte_cto_risk_metrics AS (
                                SELECT
                                a.*,
                                b.provifrs9,
                                c.tpcam,
                                d.provisio,
                                '{BDR}' AS bdr,
                                S10 + S12 + S30 + S31 + S32 + S33 + S40 + S51 + S54 + S57 + S61 AS Riesgo_dispuesto
                                FROM
                                cte_cto_raw_risk_metrics_new_inner A
                                LEFT JOIN (
                                SELECT
                                data_date_part AS feoperac,
                                s1emp,
                                contra1,
                                SUM(CASE WHEN importh < -99999999 THEN 0 ELSE importh END) AS provifrs9
    FROM
    cd_bdr_{BDR}.jm_prov_esotr
    WHERE
    data_date_part = '{fecha}'
    AND tip_impt = 2
    GROUP BY
    data_date_part,
    s1emp,
    contra1
    ) B ON A.feoperac = B.feoperac
    AND A.s1emp = B.s1emp
    AND A.contra1 = B.contra1
    LEFT JOIN (
                  SELECT
                  data_date_part AS feoperac,
                  coddiv,
                  TPCAM
                  FROM
                  cd_bdr_{BDR}.jm_tip_cambio
                  WHERE
                  data_date_part = '{fecha}'
                  AND coddiv2 = 'EUR'
                         ) C ON A.feoperac = C.feoperac
    AND A.coddiv = C.coddiv
    LEFT JOIN (
                  SELECT
                  feoperac,
                  contra1,
                  s1emp,
                  SUM(provisio) AS provisio
    FROM
    stage_ifrs9
    GROUP BY
    feoperac,
    contra1,
    s1emp
    ) D ON A.feoperac = D.feoperac
    AND A.contra1 = D.contra1
    AND A.s1emp = D.s1emp
    )
    SELECT * FROM cte_cto_risk_metrics;





