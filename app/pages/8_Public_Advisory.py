from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_CITY
from src.decision_engine import build_city_readiness_summary, readiness_to_color
from src.forecast_engine import make_ml_forecast


RISK_RANK = {
    "Nizak": 1,
    "Umjeren": 2,
    "Visok": 3,
    "Vrlo visok": 4,
}

LANG_UI = {
    "hr": {
        "page_title": "📣 Public Advisory / Citizen Guidance",
        "hero_title": "📣 Public Advisory / Citizen Guidance",
        "hero_subtitle": (
            "Javno-orijentirani modul sustava HeatSafe HR. Stranica pretvara toplinski rizik "
            "u jasne preporuke za građane, starije osobe, djecu, turiste, radnike na otvorenom "
            "i organizatore događaja."
        ),
        "city": "Odaberi grad",
        "language": "Jezik",
        "scenario": "Scenario mode",
        "temp_delta": "Promjena temperature (°C)",
        "humidity_delta": "Promjena vlage (%)",
        "wind_delta": "Promjena vjetra (m/s)",
        "advisory_level": "Javni advisory level",
        "next24": "Next 24h risk",
        "peak7d": "Next 7d peak",
        "peak_date": "Peak date",
        "high_days": "High-risk days",
        "notice_title": "Sažetak za javnost",
        "upcoming_notice": "Nadolazeća eskalacija",
        "audience_title": "Preporuke po skupinama",
        "report_title": "Public advisory export",
        "download_txt": "⬇ Download public advisory (.txt)",
        "scenario_on": "Scenario mode uključen",
        "scenario_off": "Scenario mode isključen",
        "general_header": "Javni bulletin",
        "today_message": "Za grad {city} sljedeća 24h procjenjuje se razina rizika {level}.",
        "peak_message": (
            "Najizraženiji signal u idućih 7 dana očekuje se {date} uz razinu {level} i score {score}."
        ),
        "escalation_message": (
            "Iako je neposredni javni rizik trenutno {current}, sustav očekuje rast do razine "
            "{peak} na datum {date}. Građani i posjetitelji trebaju pratiti nove obavijesti."
        ),
        "audiences": {
            "citizens": "Građani",
            "elderly": "Starije osobe",
            "children": "Djeca",
            "tourists": "Turisti",
            "outdoor_workers": "Radnici na otvorenom",
            "event_organizers": "Organizatori događaja",
        },
    },
    "en": {
        "page_title": "📣 Public Advisory / Citizen Guidance",
        "hero_title": "📣 Public Advisory / Citizen Guidance",
        "hero_subtitle": (
            "Public-facing module of HeatSafe HR. This page translates heat risk "
            "into clear recommendations for citizens, elderly people, children, tourists, "
            "outdoor workers, and event organizers."
        ),
        "city": "Select city",
        "language": "Language",
        "scenario": "Scenario mode",
        "temp_delta": "Temperature change (°C)",
        "humidity_delta": "Humidity change (%)",
        "wind_delta": "Wind change (m/s)",
        "advisory_level": "Public advisory level",
        "next24": "Next 24h risk",
        "peak7d": "Next 7d peak",
        "peak_date": "Peak date",
        "high_days": "High-risk days",
        "notice_title": "Public summary",
        "upcoming_notice": "Upcoming escalation",
        "audience_title": "Audience-based guidance",
        "report_title": "Public advisory export",
        "download_txt": "⬇ Download public advisory (.txt)",
        "scenario_on": "Scenario mode enabled",
        "scenario_off": "Scenario mode disabled",
        "general_header": "Public bulletin",
        "today_message": "For {city}, the estimated risk level for the next 24h is {level}.",
        "peak_message": (
            "The strongest signal within the next 7 days is expected on {date} with level {level} and score {score}."
        ),
        "escalation_message": (
            "Although the immediate public risk is currently {current}, the system expects a rise "
            "to {peak} on {date}. Citizens and visitors should follow new updates."
        ),
        "audiences": {
            "citizens": "Citizens",
            "elderly": "Elderly people",
            "children": "Children",
            "tourists": "Tourists",
            "outdoor_workers": "Outdoor workers",
            "event_organizers": "Event organizers",
        },
    },
}


ADVISORY_DATA = {
    "hr": {
        "citizens": {
            "Nizak": [
                "Nastavite pratiti osnovne vremenske uvjete i planirajte dnevne aktivnosti uobičajeno, ali uz osnovnu zaštitu od sunca.",
                "Pri duljem boravku vani nosite bocu vode, laganu i svijetlu odjeću te, po mogućnosti, šešir ili kapu.",
                "Ako ste duže na otvorenom, planirajte povremene pauze u hladu ili u rashlađenom prostoru.",
                "Provjerite nove gradske, turističke ili meteorološke obavijesti ako se prognoza mijenja prema toplijim uvjetima.",
                "Posebno oprezni budite ako imate kronične bolesti, uzimate terapiju koja može pojačati osjetljivost na vrućinu ili dulje putujete.",
            ],
            "Umjeren": [
                "Smanjite dulji boravak na suncu u najtoplijem dijelu dana, osobito oko podneva i ranog poslijepodneva.",
                "Pijte dovoljno vode tijekom dana i ne čekajte da osjetite jaku žeđ prije nego što se hidrirate.",
                "Birajte hlad, klimatizirane prostore i sporiji tempo kretanja kada je to moguće.",
                "Posebno obratite pažnju na simptome umora, glavobolje, crvenila kože, pojačanog znojenja i osjećaja pregrijavanja.",
                "Ako planirate fizički napor, sport ili duži hod, pokušajte ih prebaciti na rano jutro ili večer.",
            ],
            "Visok": [
                "Izbjegavajte nepotreban boravak na otvorenom između približno 11 i 17 sati, posebno na izravnom suncu i bez hlada.",
                "Planirajte obveze rano ujutro ili navečer te pravite češće pauze u rashlađenim prostorima.",
                "Povećajte unos vode i lagane hrane te izbjegavajte pojačani fizički napor, teške obroke i dugotrajno stajanje na suncu.",
                "Ako osjetite vrtoglavicu, slabost, mučninu, ubrzan puls ili jaku iscrpljenost, odmah potražite hlad, mir i rashlađivanje.",
                "Provjerite članove obitelji, susjede ili poznanike koji su stariji, bolesni ili žive sami jer su osjetljiviji na toplinski stres.",
            ],
            "Vrlo visok": [
                "Ostanite u zatvorenom i rashlađenom prostoru kad god je to moguće, osobito tijekom dana i u razdoblju najviše temperature.",
                "Odgodite fizički zahtjevne aktivnosti, kraće boravite vani samo ako je nužno i planirajte kretanje isključivo u sigurnijim terminima.",
                "Pijte vodu redovito u manjim količinama kroz dan, rashlađujte prostor i tijelo te nosite laganu odjeću.",
                "Pratite službene gradske, zdravstvene i meteorološke preporuke te se pridržavajte eventualnih posebnih upozorenja.",
                "Redovito provjeravajte starije članove obitelji, kronične bolesnike, susjede i osobe koje bi mogle imati poteškoće s rashlađivanjem prostora.",
            ],
        },
        "elderly": {
            "Nizak": [
                "Održavajte redovitu hidraciju i laganu dnevnu rutinu, čak i kada vrućina još nije izražena.",
                "Izbjegavajte dulje stajanje ili sjedenje na suncu bez potrebe, posebno ako uzimate redovitu terapiju.",
                "Držite vodu ili drugo prikladno piće pri ruci tijekom cijelog dana.",
                "Prostor u kojem boravite povremeno prozračite u hladnijem dijelu dana i po potrebi lagano rashladite.",
                "Pratite eventualne nove preporuke za toplije dane i unaprijed se pripremite ako forecast pokazuje rast rizika.",
            ],
            "Umjeren": [
                "Boravite češće u hladu ili rashlađenim prostorima, osobito oko podneva i ranog poslijepodneva.",
                "Pijte vodu redovito i ne oslanjajte se samo na osjećaj žeđi jer on kod starijih osoba može biti slabiji.",
                "Ako imate kronične bolesti ili terapiju, dodatno pripazite na umor, slabost i pregrijavanje.",
                "Nosite laganu, prozračnu odjeću i izbjegavajte nepotrebne izlaske u toplijem dijelu dana.",
                "Po mogućnosti neka član obitelji, susjed ili skrbnik povremeno provjeri kako se osjećate tijekom toplijih dana.",
            ],
            "Visok": [
                "Smanjite izlazak iz doma u najtoplijem dijelu dana na minimum i obaveze planirajte rano ujutro ili navečer.",
                "Osigurajte redovito hlađenje prostora, dostupnu vodu, laganu odjeću i mjesto za odmor u hladu ili rashlađenoj prostoriji.",
                "Ne opterećujte se težim kućanskim poslovima ili duljim hodanjem po vrućini.",
                "Članovi obitelji, susjedi ili njegovatelji trebaju češće provjeravati starije i kronične bolesnike.",
                "Kod jače slabosti, vrtoglavice, konfuzije, otežanog disanja ili naglog pogoršanja općeg stanja treba odmah potražiti pomoć.",
            ],
            "Vrlo visok": [
                "Ostanite u rashlađenom prostoru i izbjegavajte izlazak osim ako je to neizbježno.",
                "Dogovorite redovite provjere s obitelji, susjedima, njegovateljima ili prijateljima, osobito ako živite sami.",
                "Držite mobitel ili telefon pri ruci i unaprijed pripremite važne kontakte ako vam zatreba pomoć.",
                "Ako se prostor teško hladi, provedite dio dana u klimatiziranom javnom prostoru ili kod članova obitelji ako je moguće.",
                "Kod slabosti, zbunjenosti, nesigurnog hoda, otežanog disanja ili znakova toplinskog iscrpljenja odmah tražite pomoć.",
            ],
        },
        "children": {
            "Nizak": [
                "Djeca mogu boraviti vani uz redovite pauze, dovoljno vode i osnovnu zaštitu od sunca.",
                "Koristite laganu odjeću, šešir ili kapu te kremu za zaštitu od sunca prema potrebi.",
                "Organizirajte igru i boravak na otvorenom na mjestima gdje postoji dovoljno sjene.",
                "Pazite da djeca redovito piju vodu, čak i kada sama ne traže piće.",
                "Za manju djecu i bebe osigurajte zaštitu od izravnog sunca i češće provjere općeg stanja.",
            ],
            "Umjeren": [
                "Skratite intenzivnu igru na suncu u najtoplijem dijelu dana i uvodite češće odmore u hladu.",
                "Potičite djecu na uzimanje vode više puta tijekom dana, osobito nakon trčanja, igre i sporta.",
                "Planirajte sport i vanjske aktivnosti rano ujutro ili kasnije navečer kada je toplinsko opterećenje manje.",
                "Za manju djecu osigurajte stalni nadzor i rashlađivanje prostora u kojem borave.",
                "Pratite znakove umora, razdražljivosti, crvenila, glavobolje ili neuobičajene pospanosti.",
            ],
            "Visok": [
                "Izbjegavajte sportske, školske i druge zahtjevne aktivnosti na otvorenom oko podneva i u ranom poslijepodnevu.",
                "Planirajte igru i šetnje rano ujutro ili navečer te osigurajte redovitu hidraciju i odmore.",
                "Djeca trebaju češće boraviti u hladu ili rashlađenim prostorima, a boravak na suncu svesti na minimum.",
                "Posebno pazite na bebe, malu djecu i djecu s kroničnim bolestima jer su osjetljiviji na toplinski stres.",
                "Ako se jave jaka iscrpljenost, povraćanje, vrtoglavica, neobična pospanost ili visoka temperatura tijela, odmah reagirajte i potražite pomoć.",
            ],
            "Vrlo visok": [
                "Djeca trebaju većinu dana provesti u rashlađenom ili zatvorenom prostoru, a vanjske aktivnosti odgoditi.",
                "Odgodite organizirane treninge, izlete i duže igre na otvorenom ako nisu nužni.",
                "Pobrinite se da djeca redovito piju vodu, borave u laganoj odjeći i imaju mogućnost rashlađivanja prostora i tijela.",
                "Posebno zaštitite dojenčad i malu djecu od izlaganja vrućini, dugih šetnji i zagušljivih prostora.",
                "Kod znakova toplinskog stresa ne čekajte da se stanje pogorša, nego odmah maknite dijete u hlad i po potrebi potražite pomoć.",
            ],
        },
        "tourists": {
            "Nizak": [
                "Razgledavanje i aktivnosti su uglavnom sigurne uz osnovnu zaštitu od sunca i redovitu hidraciju.",
                "Uvijek nosite vodu, laganu odjeću, sunčane naočale i šešir ili kapu.",
                "Prije duljih šetnji, tura ili izleta provjerite lokalne savjete i forecast za iduće sate.",
                "Planirajte pauze u hladu, kafićima, muzejima ili drugim rashlađenim prostorima.",
                "Ako ste prvi put u gradu ili niste navikli na toplije uvjete, prilagodite tempo razgledavanja.",
            ],
            "Umjeren": [
                "Prilagodite walking ture i obilazak grada kraćim rutama, sporijim tempom i češćim pauzama.",
                "Birajte hlad, klimatizirane prostore i manje zahtjevne aktivnosti tijekom toplijeg dijela dana.",
                "Ponesite vodu i izbjegavajte duže zadržavanje na suncu oko podneva.",
                "Kod obilaska povijesnih jezgri, plaža ili otvorenih lokacija planirajte odmore unaprijed.",
                "Ako putujete sa starijim osobama ili djecom, plan dnevnih aktivnosti dodatno prilagodite toplinskim uvjetima.",
            ],
            "Visok": [
                "Premjestite intenzivnije aktivnosti, ture i izlete na rano jutro ili večer.",
                "Ograničite dulje pješačenje, čekanje u redovima i izlete bez sjene ili dostupne vode.",
                "Redovito tražite vodu, hlad i klimatizirane prostore, osobito ako se krećete kroz centar grada ili obalu.",
                "Ako osjetite vrtoglavicu, jaku žeđ, mučninu ili iscrpljenost, odmah prekinite aktivnost i sklonite se u hlad.",
                "Pratite javne preporuke i prilagodite dnevni plan sigurnijim satima i kraćim rutama.",
            ],
            "Vrlo visok": [
                "Odgodite ili snažno skratite aktivnosti na otvorenom tijekom dana i usmjerite plan na jutarnje ili večernje termine.",
                "Boravite što više u zatvorenim, rashlađenim prostorima te ograničite pješačenje i fizički napor.",
                "Ako ste na putovanju, razmotrite promjenu itinerara kako biste izbjegli najtopliji dio dana.",
                "Posebno pazite na djecu, starije osobe i članove grupe koji nisu navikli na vrućinu ili imaju zdravstvene poteškoće.",
                "Pratite lokalne preporuke i ne ignorirajte simptome pregrijavanja, posebno tijekom obilaska grada, plaže ili izleta.",
            ],
        },
        "outdoor_workers": {
            "Nizak": [
                "Nastavite rad uz standardne pauze, dovoljno vode i osnovnu zaštitu od sunca.",
                "Pratite lokalne uvjete i koristite laganu zaštitnu opremu kad god je to moguće.",
                "Planirajte hidrataciju tijekom cijele smjene, a ne tek kada se pojavi jaka žeđ.",
                "Voditelji timova trebaju pratiti forecast i unaprijed planirati eventualne prilagodbe ako toplina poraste.",
                "Na radilištu osigurajte osnovne mogućnosti sjene i pristup vodi.",
            ],
            "Umjeren": [
                "Uvedite češće pauze i veći unos vode tijekom rada na otvorenom.",
                "Kad je moguće, pomaknite teže fizičke zadatke izvan najtoplijeg dijela dana.",
                "Radnicima osigurajte sjenu, odmor i jasne upute za rano prepoznavanje toplinskog stresa.",
                "Voditelji timova trebaju pratiti znakove iscrpljenosti, vrtoglavice, usporenosti ili neuobičajenog ponašanja.",
                "Razmotrite kraće intervale rada i češće odmore za fizički zahtjevne poslove.",
            ],
            "Visok": [
                "Smanjite intenzitet rada u podnevnim i ranim popodnevnim satima te prioritet dajte sigurnijim terminima.",
                "Organizirajte sjenu, rashlađivanje i strogo planirane pauze za vodu i odmor.",
                "Voditelji timova trebaju aktivno pratiti stanje radnika i rano reagirati na znakove pregrijavanja.",
                "Ograničite radne zadatke koji traže teže nošenje, duga izlaganja suncu ili rad bez dostupne sjene.",
                "Na svakoj lokaciji trebaju biti jasne upute što učiniti ako se pojave simptomi toplinskog stresa.",
            ],
            "Vrlo visok": [
                "Odaberite samo nužne zadatke na otvorenom i odgodite ostale ako je to moguće bez ugrožavanja sigurnosti.",
                "Značajno pojačajte pauze, hlađenje, rotaciju radnika i nadzor fizičkog stanja.",
                "Rad u najtoplijem dijelu dana treba svesti na minimum ili ga, gdje god je moguće, prekinuti.",
                "Pri prvim znakovima toplinskog stresa rad treba odmah prekinuti i osobu skloniti na sigurno, u hlad ili rashlađeni prostor.",
                "Voditelji i poslodavci trebaju osigurati operativni plan za rad u uvjetima vrlo visokog toplinskog rizika.",
            ],
        },
        "event_organizers": {
            "Nizak": [
                "Događaji se mogu održavati uz osnovnu dostupnost vode, hlada i standardne informativne preporuke.",
                "Komunicirajte sudionicima osnovne savjete za boravak na otvorenom i potrebu za hidracijom.",
                "Provjerite forecast prije samog početka događaja i budite spremni na manje prilagodbe.",
                "Osigurajte osnovnu logistiku za odmor u hladu i dostupnost pitke vode.",
                "Osoblje i volonteri trebaju znati kome se javiti ako se sudionik požali na vrućinu ili slabost.",
            ],
            "Umjeren": [
                "Pojačajte dostupnost vode, hlada i kratkih zona za odmor na lokaciji događaja.",
                "Razmotrite prilagodbu trajanja i rasporeda aktivnosti na otvorenom, osobito oko podneva.",
                "Sudionicima unaprijed pošaljite savjete o vrućini, odjeći, vodi i planiranju dolaska.",
                "Povećajte broj točaka na kojima ljudi mogu kratko sjesti, rashladiti se ili predahnuti.",
                "Pratite forecast i budite spremni na dodatne mjere ako signal prijeđe u višu razinu rizika.",
            ],
            "Visok": [
                "Prilagodite termin ili raspored kako bi se izbjegao najtopliji dio dana i prevelika izloženost sudionika.",
                "Osigurajte dodatnu medicinsku, sigurnosnu i logističku spremnost na lokaciji.",
                "Smanjite fizički zahtjevne segmente, skratite boravak na otvorenom i povećajte broj rashladnih točaka.",
                "Komunicirajte sudionicima što trebaju ponijeti, kada doći i gdje se mogu rashladiti ili potražiti pomoć.",
                "Pratite broj ljudi, gustoću okupljanja i stanje najosjetljivijih skupina na lokaciji.",
            ],
            "Vrlo visok": [
                "Ozbiljno razmotrite odgodu, skraćivanje ili veliku prilagodbu događaja, osobito ako traje više sati i održava se na otvorenom.",
                "Ako se događaj ipak održava, potrebna je pojačana sigurnosna, medicinska i logistička priprema.",
                "Sudionicima i osoblju unaprijed i jasno komunicirajte visoki toplinski rizik, pravila ponašanja i plan pomoći.",
                "Povećajte broj točaka za vodu, hlad, sjedenje i brzu intervenciju te ograničite fizički zahtjevne aktivnosti.",
                "Organizator treba imati jasan plan postupanja kod toplinskog stresa, uključujući prekid ili skraćivanje programa ako se uvjeti pogoršaju.",
            ],
        },
    },
    "en": {
        "citizens": {
            "Nizak": [
                "Continue normal daily activities while monitoring basic weather conditions and using standard sun protection.",
                "Carry water, light clothing, and basic sun protection during longer outdoor stays.",
                "If you stay outside for longer periods, plan short breaks in the shade or in cooled indoor spaces.",
                "Check city, tourism, or weather updates if the forecast starts shifting toward warmer conditions.",
                "Be extra cautious if you have chronic conditions, take medication that may increase heat sensitivity, or travel long distances.",
            ],
            "Umjeren": [
                "Reduce long exposure to direct sunlight during the hottest part of the day, especially around midday and early afternoon.",
                "Drink enough water throughout the day and do not wait until you feel very thirsty.",
                "Choose shade, cooled indoor spaces, and a slower pace whenever possible.",
                "Watch for symptoms such as headache, fatigue, flushed skin, heavy sweating, or feeling overheated.",
                "If you plan exercise, sports, or long walks, move them to early morning or later evening hours.",
            ],
            "Visok": [
                "Avoid unnecessary outdoor exposure roughly between 11 AM and 5 PM, especially in direct sun and without shade.",
                "Plan errands early in the morning or later in the evening and take more frequent breaks in cooled spaces.",
                "Increase water intake, eat lighter meals, and avoid demanding physical effort or prolonged exposure to heat.",
                "If you feel dizzy, weak, nauseous, or unusually exhausted, move to shade, cool down, and rest immediately.",
                "Check on family members, neighbors, and vulnerable people who may have difficulty coping with the heat.",
            ],
            "Vrlo visok": [
                "Stay indoors in a cooled environment whenever possible, especially during daytime and peak heat hours.",
                "Postpone physically demanding activities and keep outdoor time to an absolute minimum.",
                "Drink water regularly in small amounts throughout the day and cool both your body and your surroundings.",
                "Follow official city, health, and weather guidance and pay attention to any urgent public warnings.",
                "Regularly check on older people, chronically ill individuals, neighbors, and others who may be at higher risk.",
            ],
        },
        "elderly": {
            "Nizak": [
                "Maintain regular hydration and a light daily routine even when the heat level is still relatively low.",
                "Avoid unnecessary long exposure to direct sunlight, especially if you are on regular medication.",
                "Keep water or another suitable drink close to you throughout the day.",
                "Ventilate and cool your living space during cooler hours whenever possible.",
                "Follow updates if the forecast shows a possible rise in heat risk in the coming days.",
            ],
            "Umjeren": [
                "Spend more time in shade or cooled indoor spaces, especially around midday and early afternoon.",
                "Drink water regularly and do not rely only on feeling thirsty, as thirst can be weaker in older adults.",
                "If you have chronic conditions, pay extra attention to tiredness, weakness, and signs of overheating.",
                "Wear light, breathable clothing and reduce non-essential outdoor activity during warmer hours.",
                "If possible, ask a family member, neighbor, or caregiver to check on you from time to time during warm days.",
            ],
            "Visok": [
                "Reduce leaving home during the hottest part of the day to the minimum and move errands to safer hours.",
                "Ensure cooling, light clothing, easy access to drinking water, and a shaded or cooled resting area.",
                "Avoid demanding housework, long walks, and unnecessary physical effort in the heat.",
                "Family members, neighbors, or caregivers should check more often on older or chronically ill people.",
                "If weakness, confusion, dizziness, breathing problems, or a sudden decline in condition appear, seek help immediately.",
            ],
            "Vrlo visok": [
                "Remain in a cooled indoor space and avoid going out unless it is absolutely necessary.",
                "Arrange regular check-ins with family, neighbors, caregivers, or friends, especially if you live alone.",
                "Keep your phone nearby and prepare important contact numbers in advance in case you need help.",
                "If your home is difficult to cool, spend part of the day in an air-conditioned public space or with relatives if possible.",
                "If weakness, confusion, unstable walking, breathing difficulties, or signs of heat stress appear, seek help without delay.",
            ],
        },
        "children": {
            "Nizak": [
                "Children can stay outdoors with regular breaks, water, and basic sun protection.",
                "Use light clothing, hats, and appropriate sunscreen when needed.",
                "Plan outdoor play in areas where shade is available.",
                "Make sure children drink water regularly, even if they do not ask for it.",
                "For babies and younger children, provide close supervision and protection from direct sunlight.",
            ],
            "Umjeren": [
                "Shorten intense outdoor play during the hottest part of the day and add more breaks in the shade.",
                "Encourage children to drink water more often, especially after running, sports, or active play.",
                "Plan sports and outdoor activity early in the morning or later in the evening.",
                "For younger children, ensure supervision and access to cooled indoor spaces.",
                "Watch for signs of tiredness, irritability, flushed skin, headache, or unusual sleepiness.",
            ],
            "Visok": [
                "Avoid sports, school, and other demanding outdoor activities around midday and early afternoon.",
                "Schedule play and walks early in the morning or later in the evening with frequent hydration.",
                "Children should spend more time in shade or cooled indoor spaces and minimize sun exposure.",
                "Pay special attention to babies, small children, and children with chronic illnesses, as they are more vulnerable to heat stress.",
                "If severe fatigue, vomiting, dizziness, unusual drowsiness, or high body temperature occur, react immediately and seek help.",
            ],
            "Vrlo visok": [
                "Children should spend most of the day in cooled or indoor spaces and outdoor activities should be postponed.",
                "Postpone organized sports, excursions, and long outdoor play unless truly necessary.",
                "Make sure children drink water regularly, wear light clothing, and stay in a cooled environment.",
                "Provide extra protection for infants and very young children from heat, long walks, and poorly ventilated spaces.",
                "Do not wait for symptoms to worsen; if signs of heat stress appear, move the child to a cool place immediately and seek help if needed.",
            ],
        },
        "tourists": {
            "Nizak": [
                "Sightseeing and outdoor activities are generally safe with basic sun protection and regular hydration.",
                "Always carry water, light clothing, sunglasses, and a hat.",
                "Before longer walks, guided tours, or excursions, check local advice and the forecast for the coming hours.",
                "Plan short breaks in the shade, cafés, museums, or other cooled indoor spaces.",
                "If you are not used to warmer climates, adjust the pace of your sightseeing accordingly.",
            ],
            "Umjeren": [
                "Adapt walking tours and sightseeing plans with shorter routes, a slower pace, and more frequent breaks.",
                "Choose shade, cooled indoor places, and less demanding activities during warmer hours.",
                "Carry water and avoid long periods in direct sun around midday.",
                "When visiting historic centers, beaches, or open areas, plan rest stops in advance.",
                "If you are traveling with children, older adults, or vulnerable companions, adapt the daily plan more carefully.",
            ],
            "Visok": [
                "Move more demanding activities, tours, and excursions to early morning or evening hours.",
                "Limit long walks, long queues, and excursions without access to shade or drinking water.",
                "Regularly seek water, shade, and air-conditioned spaces, especially in city centers or on the coast.",
                "If you feel dizzy, very thirsty, nauseous, or exhausted, stop the activity immediately and move to a cooler place.",
                "Follow public guidance and adjust your schedule to safer hours and shorter routes.",
            ],
            "Vrlo visok": [
                "Postpone or significantly reduce outdoor activities during daytime and focus on early morning or late evening plans.",
                "Stay as much as possible in cooled indoor environments and limit walking and physical strain.",
                "If you are traveling, consider adjusting your itinerary to avoid the hottest part of the day.",
                "Pay special attention to children, older adults, and group members who are not used to heat or have health conditions.",
                "Follow local public guidance closely and do not ignore signs of overheating during sightseeing, beach visits, or excursions.",
            ],
        },
        "outdoor_workers": {
            "Nizak": [
                "Continue work with standard breaks, hydration, and sun protection.",
                "Monitor local conditions and use light protective equipment whenever possible.",
                "Plan hydration throughout the shift rather than waiting until strong thirst appears.",
                "Team leads should monitor the forecast and be ready to adapt work if temperatures rise.",
                "Ensure that basic shade and water access are available at the work location.",
            ],
            "Umjeren": [
                "Introduce more frequent breaks and increased water intake during outdoor work.",
                "When possible, move heavier tasks away from the hottest hours of the day.",
                "Provide workers with shade, rest, and clear instructions on recognizing early heat stress symptoms.",
                "Supervisors should monitor for exhaustion, dizziness, slowed reactions, or unusual behavior.",
                "Consider shorter work intervals and more frequent rest periods for physically demanding tasks.",
            ],
            "Visok": [
                "Reduce work intensity during midday and early afternoon and prioritize safer working hours.",
                "Ensure shade, cooling, and strictly scheduled hydration and rest breaks.",
                "Supervisors should actively monitor workers’ condition and respond early to signs of overheating.",
                "Limit tasks that involve heavy lifting, prolonged sun exposure, or work without access to shade.",
                "Each site should have clear instructions on what to do if heat stress symptoms occur.",
            ],
            "Vrlo visok": [
                "Perform only essential outdoor tasks and postpone the rest whenever this can be done safely.",
                "Significantly increase rest periods, cooling measures, worker rotation, and physical monitoring.",
                "Work during peak heat hours should be reduced to the minimum or suspended whenever possible.",
                "At the first signs of heat stress, stop work immediately and move the person to shade or a cooled area.",
                "Supervisors and employers should have an operational plan specifically for very high heat-risk working conditions.",
            ],
        },
        "event_organizers": {
            "Nizak": [
                "Events can proceed with basic water, shade, and standard public guidance.",
                "Communicate basic heat and hydration advice to participants.",
                "Review the forecast again shortly before the event begins.",
                "Ensure basic logistics for short rest in the shade and access to drinking water.",
                "Staff and volunteers should know who to contact if a participant feels unwell due to heat.",
            ],
            "Umjeren": [
                "Increase water, shade, and short rest-zone availability across the venue.",
                "Consider adjusting the duration and timing of outdoor activities, especially around midday.",
                "Send participants advance guidance on heat, clothing, hydration, and arrival planning.",
                "Increase the number of points where people can briefly sit, cool down, or rest.",
                "Monitor the forecast and be prepared to add further measures if the risk rises.",
            ],
            "Visok": [
                "Adjust the schedule to avoid the hottest part of the day and reduce prolonged exposure.",
                "Ensure additional medical, safety, and logistical readiness at the venue.",
                "Reduce physically demanding segments, shorten outdoor exposure, and increase cooling points.",
                "Clearly communicate what participants should bring, when to arrive, and where to find water, shade, and assistance.",
                "Monitor crowd density, participant flow, and the condition of the most vulnerable groups on site.",
            ],
            "Vrlo visok": [
                "Seriously consider postponing, shortening, or heavily adapting the event, especially if it lasts for hours outdoors.",
                "If the event proceeds, stronger safety, medical, and logistical readiness is required.",
                "Clearly communicate the high heat risk, safety rules, and help procedures to staff and participants in advance.",
                "Increase the number of water, shade, seating, and rapid-response points, and reduce physically demanding activities.",
                "The organizer should have a clear contingency plan for heat stress, including interruption or shortening of the program if conditions worsen.",
            ],
        },
    },
}


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str) -> None:
    st.markdown(
        f'<span class="status-pill" style="background:{color};">{text}</span>',
        unsafe_allow_html=True,
    )


def render_list_card(title: str, items: list[str]) -> None:
    list_html = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="panel-title">{title}</div>
            <ul class="soft-list">
                {list_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def level_color(level: str) -> str:
    return {
        "Nizak": "#2E8B57",
        "Umjeren": "#E6A700",
        "Visok": "#E67E22",
        "Vrlo visok": "#C0392B",
    }.get(level, "#64748b")


def build_public_brief(
    city: str,
    lang: str,
    summary: dict,
    scenario_enabled: bool,
    temperature_delta: float,
    humidity_delta: float,
    wind_delta: float,
) -> str:
    ui = LANG_UI[lang]
    peak_date = pd.to_datetime(summary["next_7d_peak_date"]).strftime("%d.%m.%Y.")

    scenario_line = ui["scenario_on"] if scenario_enabled else ui["scenario_off"]
    if scenario_enabled:
        scenario_line += (
            f" | ΔT {temperature_delta:+.1f} °C | ΔRH {humidity_delta:+.1f}% | ΔWind {wind_delta:+.1f} m/s"
        )

    parts = [
        "HEATSAFE HR — PUBLIC ADVISORY",
        "",
        f"City / Grad: {city}",
        scenario_line,
        "",
        f"{ui['next24']}: {summary['next_24h_level']} ({summary['next_24h_score']:.1f})",
        f"{ui['peak7d']}: {summary['next_7d_peak_level']} ({summary['next_7d_peak_score']:.1f})",
        f"{ui['peak_date']}: {peak_date}",
        f"{ui['high_days']}: {summary['high_risk_days']}",
        "",
    ]

    for audience_key, audience_title in ui["audiences"].items():
        parts.append(audience_title.upper())
        for item in ADVISORY_DATA[lang][audience_key][summary["next_24h_level"]]:
            parts.append(f"- {item}")
        parts.append("")

    return "\n".join(parts).strip()


st.markdown(
    f"""
    <div class="page-hero">
        <div class="page-hero-title">{LANG_UI['hr']['hero_title']}</div>
        <div class="page-hero-subtitle">{LANG_UI['hr']['hero_subtitle']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

cities = [
    "Dubrovnik",
    "Osijek",
    "Rijeka",
    "Split",
    "Šibenik",
    "Zadar",
    "Zagreb",
]

lang_col, city_col, scenario_col = st.columns([1, 1.2, 1])

with lang_col:
    lang_choice = st.radio(
        "Language / Jezik",
        options=["Hrvatski", "English"],
        horizontal=True,
    )
    lang = "hr" if lang_choice == "Hrvatski" else "en"

ui = LANG_UI[lang]

default_city = st.session_state.get("selected_city", DEFAULT_CITY)
default_index = cities.index(default_city) if default_city in cities else 0

with city_col:
    selected_city = st.selectbox(ui["city"], cities, index=default_index)
    st.session_state.selected_city = selected_city

with scenario_col:
    scenario_enabled = st.toggle(ui["scenario"], value=True)

if scenario_enabled:
    s1, s2, s3 = st.columns(3)
    with s1:
        temperature_delta = st.slider(ui["temp_delta"], -2, 12, 6, 1)
    with s2:
        humidity_delta = st.slider(ui["humidity_delta"], -20, 30, 10, 1)
    with s3:
        wind_delta = st.slider(ui["wind_delta"], -8, 5, -3, 1)
else:
    temperature_delta = 0
    humidity_delta = 0
    wind_delta = 0

try:
    active_df = make_ml_forecast(
        selected_city,
        temperature_delta=temperature_delta,
        humidity_delta=humidity_delta,
        wind_delta=wind_delta,
    )
except Exception as exc:
    st.error(f"Public advisory nije dostupan: {exc}")
    st.stop()

summary = build_city_readiness_summary(selected_city, active_df)

st.markdown(f"## {ui['advisory_level']}")
badge(summary["next_24h_level"], level_color(summary["next_24h_level"]))

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card(ui["next24"], summary["next_24h_level"], f"{summary['next_24h_score']:.1f}")
with k2:
    metric_card(ui["peak7d"], summary["next_7d_peak_level"], f"{summary['next_7d_peak_score']:.1f}")
with k3:
    metric_card(ui["peak_date"], pd.to_datetime(summary["next_7d_peak_date"]).strftime("%d.%m.%Y."))
with k4:
    metric_card(ui["high_days"], str(summary["high_risk_days"]), summary["readiness_status"])

st.info(
    ui["today_message"].format(
        city=selected_city,
        level=summary["next_24h_level"],
    )
)

st.markdown(
    f"""
    <div class="note-box">
        <b>{ui['general_header']}:</b><br>
        {ui['peak_message'].format(
            date=pd.to_datetime(summary['next_7d_peak_date']).strftime('%d.%m.%Y.'),
            level=summary['next_7d_peak_level'],
            score=f"{summary['next_7d_peak_score']:.1f}",
        )}
    </div>
    """,
    unsafe_allow_html=True,
)

if RISK_RANK[summary["next_7d_peak_level"]] > RISK_RANK[summary["next_24h_level"]]:
    st.markdown(
        f"""
        <div class="warning-box">
            <b>{ui['upcoming_notice']}:</b><br>
            {ui['escalation_message'].format(
                current=summary['next_24h_level'],
                peak=summary['next_7d_peak_level'],
                date=pd.to_datetime(summary['next_7d_peak_date']).strftime('%d.%m.%Y.'),
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(f"## {ui['audience_title']}")

audience_titles = ui["audiences"]

row1 = st.columns(3)
with row1[0]:
    render_list_card(
        audience_titles["citizens"],
        ADVISORY_DATA[lang]["citizens"][summary["next_24h_level"]],
    )
with row1[1]:
    render_list_card(
        audience_titles["elderly"],
        ADVISORY_DATA[lang]["elderly"][summary["next_24h_level"]],
    )
with row1[2]:
    render_list_card(
        audience_titles["children"],
        ADVISORY_DATA[lang]["children"][summary["next_24h_level"]],
    )

row2 = st.columns(3)
with row2[0]:
    render_list_card(
        audience_titles["tourists"],
        ADVISORY_DATA[lang]["tourists"][summary["next_24h_level"]],
    )
with row2[1]:
    render_list_card(
        audience_titles["outdoor_workers"],
        ADVISORY_DATA[lang]["outdoor_workers"][summary["next_24h_level"]],
    )
with row2[2]:
    render_list_card(
        audience_titles["event_organizers"],
        ADVISORY_DATA[lang]["event_organizers"][summary["next_24h_level"]],
    )

brief_text = build_public_brief(
    city=selected_city,
    lang=lang,
    summary=summary,
    scenario_enabled=scenario_enabled,
    temperature_delta=temperature_delta,
    humidity_delta=humidity_delta,
    wind_delta=wind_delta,
)

st.markdown('<div class="report-box">', unsafe_allow_html=True)
st.markdown(f"### {ui['report_title']}")
st.download_button(
    ui["download_txt"],
    data=brief_text.encode("utf-8"),
    file_name=f"heatsafe_hr_public_advisory_{selected_city}_{lang}.txt",
    mime="text/plain",
    use_container_width=True,
    key=f"dl_public_advisory_{selected_city}_{lang}",
)
st.code(brief_text, language="text")
st.markdown("</div>", unsafe_allow_html=True)