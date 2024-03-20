Create table default.perimetro_{pais}_{aaaamm} as
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
    l3 not in ('304') and
    l4 not in ('3022','3021') and
    l5 not in ('30002','30012','30022','30032','30302','30392') and
    l2 in ('30')
    then importh
    else 0
    end as S12
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
    l5 in ('14022','14032','14042') or
    l7 in ('1405012')
    then importh
    else 0
    end as S40
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
    from cd_bdr_{BDR}.jm_posic_contr A
    where data_date_part='{fecha}'
    ) A
    ),

    saldos_contra as
(
    select	feoperac
    ,  s1emp
    , contra1
    , sum(S10) as S10
    , sum(S12) as S12
    , sum(S30) as S30
    , sum(S31) as S31
    , sum(S32) as S32
    , sum(S33) as S33
    , sum(S40) as S40
    , sum(S51) as S51
    , sum(S54) as S54
    , sum(S57) as S57
    , sum(S61) as S61
    from saldos
    group by
    feoperac
    ,  s1emp
    , contra1
    ),

    stage_IFRS9 as
(
    select  feoperac,s1emp,contra1,stage,case when provisio < -99999999 then 0 else provisio end as provisio
    from cd_bdr_{BDR}.jm_cto_ifrs9
    where data_date_part='{fecha}' and tajifrs9 = 1
                ),

    catego as
(
    select  a.feoperac
    , a.s1emp
    , a.contra1
    , a.id_prod
    , a.idnumcli
    , (sum(coalesce(on_balance_reg,0))) as on_balance_reg
    , (sum(coalesce(off_balance_reg,0))) as off_balance_reg
    --    , (sum(coalesce(on_balance_reg_old,0))) as on_balance_reg_old
    --    , (sum(coalesce(off_balance_reg_old,0))) as off_balance_reg_old
    , sum(coalesce(exp_orig,0)) as exp_orig
    , sum(coalesce(ead_reg,0)) as ead_reg
    , sum(coalesce(rwa_reg,0)) as rwa_reg
    , sum(coalesce(provdbal,0)) as provdbal
    , sum(coalesce(PROVFBAL,0)) as PROVFBAL
    , sum(coalesce(provisiones,0)) as provisiones
    --	, approach
    , sum(coalesce(b.provisio,0)) as provisio
    , cli_type_id
    , SECURED


    from
(
    select A.feoperac
    , A.s1emp
    , A.contra1
    , a.idnumcli
    , id_prod
    --        , sum(coalesce(decode(ind_mmff,'1',0,-isdebala),0)) as on_balance_reg_old
    --        , sum(coalesce(decode(ind_mmff,'1',0,-isfubala),0)) as off_balance_reg_old
    , sum(coalesce(decode(ind_mmff,'1',eadfinal, (abs(LEAST(isdebala,0)))),0)) as on_balance_reg
    , sum(coalesce(decode(ind_mmff,'1',0, (abs(LEAST(isfubala,0)))),0)) as off_balance_reg
    , sum(coalesce(exp_orig,0)) as exp_orig
    , sum(coalesce(eadfinal,0)) as ead_reg
    , sum(coalesce(criavesc,0))*12.5 as rwa_reg
    , sum(coalesce(provdbal,0)) as provdbal
    , sum(coalesce(PROVFBAL,0)) as PROVFBAL
    , sum(provdbal) + sum(PROVFBAL) as provisiones
    --       ,"IRB" as approach
    , tip_pert as cli_type_id
    ,case  WHEN nvl(ind_secured_re_ctr,0)=1  THEN 1
    ELSE 0 END AS SECURED


    from cd_bdr_{BDR}.jm_cto_aj_reg A
    left join
(select ctr_gar.feoperac,ctr_gar.s1emp,ctr_gar.contra1, case when sum(nvl(val_gar.imgarant,0))>0 then 1 else 0 end ind_secured_re_ctr
    from cd_bdr_{BDR}.jm_garant_cto ctr_gar
    left join
    cd_bdr_{BDR}.jm_val_gara val_gar
    on  val_gar.feoperac=ctr_gar.feoperac and
    val_gar.s1emp=ctr_gar.s1emp and
    val_gar.biengar1=ctr_gar.biengar1
    where
    ctr_gar.tip_gara=2 and
    ctr_gar.cod_gar in (11,25,81,82,83,84,85,86,87,88,89,90) AND
    ctr_gar.feoperac = '{fecha}'
    group by ctr_gar.feoperac,
    ctr_gar.s1emp,
    ctr_gar.contra1 ) gar

    on gar.s1emp=A.s1emp and gar.feoperac='{fecha}' and gar.contra1=A.contra1
    where data_date_part='{fecha}'

    and
(
    (motexinf in (0) or motexinf is null) or
(motexinf in (20021,20023,19) and cat_msta in(8,17,28,30) and c_rw_irb = 0)
    )
    group by feoperac
    , s1emp
    , contra1
    , id_prod
    ,idnumcli
    --    , approach
    ,cli_type_id
    ,SECURED

    union all
    select
    feoperac
    , a.s1emp
    , a.contra1
    , a.idnumcli
    , id_prod
    --        , sum(coalesce(decode(flg_mmff,1,0,-sdbpr),0)) as on_balance_reg_old
    --        , sum(coalesce(decode(flg_mmff,1,0,-sfbpr),0)) as off_balance_reg_old
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
    --      ,"STD" as approach
    ,tipersti as cli_type_id
    ,coalesce (secured,0) as  secured

    from cd_bdr_{BDR}.jm_amp_aj_std A
    left join
(select s1emp,contra1, case when sbcprfin in (4,5) THEN 1 ELSE 0 END AS SECURED
    from cd_bdr_{BDR}.jm_amp_aj_std
    where data_date_part='{fecha}' and sbcprfin in (4,5) AND motexinf !=50065)B
    on a.s1emp=b.s1emp and a.contra1=b.contra1

    where data_date_part='{fecha}'

    and
(
    motexinf !=50065
)
    group by feoperac
    , s1emp
    , contra1
    , id_prod
    ,idnumcli
    --   , approach
    , cli_type_id
    , SECURED

    ) A
    left JOIN (select feoperac,contra1,s1emp,stage,sum(provisio) as provisio from stage_ifrs9 group by feoperac,contra1,s1emp,stage ) B
    on a.feoperac = b.feoperac and a.contra1=b.contra1 and a.s1emp=b.s1emp

    group by a.feoperac
    , a.s1emp
    , a.contra1
    , a.id_prod
    , a.idnumcli
    --  , approach
    , cli_type_id
    , SECURED
    ),

    catego2 as
(
    Select            A.feoperac
    , A.s1emp
    , A.contra1
    , a.idnumcli
    , id_prod
    , sum(on_balance_reg) as on_balance_reg
    , sum(off_balance_reg) as off_balance_reg
    , sum(exp_orig) as exp_orig
    , sum(ead_reg) as ead_reg
    , sum(provdbal) as provdbal
    , sum(PROVFBAL) as PROVFBAL
    , sum(provisiones) as provisiones
    , cli_type_id
    , b.secured
    , clisegm

    from catego A
    left join
(select contra1,s1emp,SECURED
 from catego
 where secured in (1)) B
    on a.s1emp=b.s1emp and a.contra1=b.contra1
    left join
(select feoperac,s1emp ,idnumcli, clisegm
 from cd_bdr_{BDR}.jm_client_bii
 where data_date_part = '{fecha}' ) C
    on a.s1emp=c.s1emp and a.idnumcli=c.idnumcli and a.feoperac = c.feoperac

    group by          A.feoperac
    , A.s1emp
    , A.contra1
    , id_prod
    , cli_type_id
    ,a.idnumcli
    , b.secured
    ,clisegm

    ),

    cto_raw_risk_metrics_new_inner as
(
    select    a.feoperac
    , a.s1emp
    , a.contra1
    ,b.idnumcli
    , stage
    , id_prod
    , cli_type_id
    , secured
    ,clisegm
    , '{divisa}' as coddiv
    , case when b.contra1 is null then 0 else 1 end as perimetro_catego
    , sum(on_balance_reg) as on_balance_reg
    , sum(off_balance_reg) as off_balance_reg
    , sum(exp_orig) as exp_orig
    , sum(ead_reg) as ead_reg
    , sum(provdbal) as provdbal
    , sum(PROVFBAL) as PROVFBAL
    , sum(provisiones) as provisiones
    --	    , sum(coalesce(b.provisio,0)) as provisio
    , sum(S10) as S10
    , sum(S12) as S12
    , sum(S30) as S30
    , sum(S31) as S31
    , sum(S32) as S32
    , sum(S33) as S33
    , sum(S40) as S40
    , sum(S51) as S51
    , sum(S54) as S54
    , sum(S57) as S57
    , sum(S61) as S61
    , cast(from_unixtime(unix_timestamp(), 'yyyyMMddHHmmss') as bigint) as process_timestamp

    from saldos_contra A
    full join catego2 B on
    A.feoperac = B.feoperac
    and A.s1emp=B.s1emp
    and A.contra1=B.contra1
    left JOIN (select distinct feoperac,contra1,s1emp,stage,provisio from stage_ifrs9) C
    on a.feoperac = C.feoperac and a.contra1=C.contra1 and a.s1emp=C.s1emp
    group by
    A.feoperac
    , A.s1emp
    , A.contra1
    , perimetro_catego
    , B.ID_PROD
    , C.stage
    , coddiv
    , secured
    , cli_type_id
    ,b.idnumcli
    ,clisegm
    ),

    cto_risk_metrics as
(
    select  a.*,b.provifrs9, c.tpcam,D.provisio,'{BDR}' as bdr

    , S10 + S12
    + S30 + S31 + S32 + S33
    + S40
    + S51 + S54 + S57
    + S61
    as Riesgo_dispuesto


    from cto_raw_risk_metrics_new_inner A
    left join
    (
    select data_date_part as feoperac, s1emp, contra1, sum(case when importh < -99999999 then 0 else importh end) as provifrs9
    from cd_bdr_{BDR}.jm_prov_esotr
    where data_date_part='{fecha}'
    and tip_impt=2
    group by data_date_part, s1emp, contra1
    ) B
    on
    A.feoperac = B.feoperac
    and	A.s1emp=B.s1emp
    and A.contra1=B.contra1
    left join
(
    select data_date_part as feoperac,coddiv,TPCAM
    from cd_bdr_{BDR}.jm_tip_cambio
    where data_date_part='{fecha}' and coddiv2 = 'EUR'
                ) C
    on
    a.feoperac=c.feoperac
    and A.coddiv=C.coddiv

    left JOIN (select feoperac,contra1,s1emp,sum(provisio) as provisio from stage_ifrs9 group by feoperac,contra1,s1emp ) D
    on a.feoperac = D.feoperac and a.contra1=D.contra1 and a.s1emp=D.s1emp
    )

select *
from cto_risk_metrics