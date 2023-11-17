### WEAT words and expansions
from copy import copy

from wefe.datasets.datasets import load_weat
from wefe.query import Query

career = ["executive", "management", "professional", "corporation", "salary", "office",
          "business", "career"]
career_exp = ['department', 'offices', 'employment', 'corporations', 'industry', 'subsidiary',
              'expert', 'manage', 'paid', 'stint', 'success', 'corp', 'wages', 'vice', 'wage',
              'entity', 'staff', 'desk', 'income', 'enterprise', 'careers', 'managers',
              'marketing', 'chief', 'companies', 'successful', 'job', 'accounting',
              'vice-president', 'appointed', 'services', 'headquarters', 'corp.', 'managment',
              'appointment', 'competent', 'salaries', 'development', 'professionals',
              'corporate', 'earning', 'president', 'knowledgeable', 'coaching',
              'professionally', 'executives', 'qualified', 'skilled', 'subsidiaries', 'pay',
              'director', 'experienced', 'businesses', 'managing', 'payroll', 'consulting',
              'company']
all_career = career + career_exp

family = ["home", "parents", "children", "family", "cousins", "marriage", "wedding",
          "relatives"]
family_exp = ['houses', 'house', 'friends', 'kids', 'siblings', 'babies',
              'marital', 'families', 'in-laws', 'apartment', 'prom', 'toddlers', 'marry',
              'weddings', 'grandparents', 'infants', 'cousin', 'child', 'nieces']
all_family = family + family_exp

art = ['art', 'poetry', 'symphony', 'drama', 'dance', 'Shakespeare', 'sculpture', 'literature',
       'novel']
art_exp = ['symphonic', 'romance', 'adaptation', 'shakespearean', 'poetic', 'dances',
           'philharmonic', 'prose', 'music', 'choral', 'poems', 'artworks', 'orchestral',
           'artistic', 'poem', 'sculptures', 'arts', 'novels', 'melodrama', 'choreography',
           'literary', 'fiction', 'novella', 'paintings', 'dancers', 'painting']
all_art = art + art_exp

math_sci = ['chemistry', 'math', 'science', 'computation', 'Einstein', 'experiment', 'algebra',
             'geometry', 'NASA', 'astronomy', 'physics', 'numbers', 'technology', 'equations',
             'addition', 'calculus']
math_sci_exp = ['innovations', 'sciences', 'biochemistry', 'technologies', 'astrophysics',
                'algorithms', 'experimenting', 'algorithm', 'nonlinear', 'computations',
                'astronomers', 'quadratic', 'arithmetic', 'experimental', 'astronomical',
                'digits', 'counting', 'telescope', 'count', 'research', 'molecular', 'innovation',
                'innovative', 'idea', 'chemical', 'technological', 'biology', 'scientific',
                'mathematical', 'geometric', 'high-tech', 'symmetry', 'telescopes',
                'approximation', 'astronomer', 'numerical', 'laboratory', 'computational',
                'maths', 'cosmology', 'microbiology', 'quantum', 'geometries', 'equation',
                'number', 'theory', 'compute', 'experimented', 'geometrical', 'percentages',
                'percentage', 'trigonometry', 'relativity', 'experimentation', 'algebraic',
                'mathematics', 'experiments', 'calculations', 'calculation']

all_math_sci = math_sci + math_sci_exp

male = ['grandfather', 'uncle', 'son', 'boy', 'father', 'he', 'him', 'his', 'man', 'male',
        'brother']
male_exp = ['guy', 'himself', 'nephew', 'grandson', 'men', 'boys', 'great-grandfather',
            'father-in-law', 'husband', 'brothers', 'males', 'brother-in-law', 'sons', 'dad']
all_male = male + male_exp

female = ['daughter', 'she', 'her', 'grandmother', 'mother', 'aunt', 'sister', 'hers', 'woman',
          'female', 'girl']
female_exp = ['grandma', 'herself', 'sister-in-law', 'niece', 'sisters', 'mom', 'mother-in-law',
              'lady', 'wife', 'females', 'girls', 'great-grandmother', 'women', 'sexy',
              'granddaughter', 'daughters']

all_female = female + female_exp


#### Spanish ###
es_math_sci= ["científico", "físico", "químico", "astrónomo", "tecnológico", "biólogo", "científica",
           "física", "química", "astrónoma", "tecnológica", "bióloga"]
es_math_sci_exp = ['toxicológico', 'neurocientífico', 'enzimático', 'inorgánico', 'indólogo', 'cosmógrafo', \
                  'antropólogo', 'inorgánica', 'fitoquímica', 'cognitivo', 'ornitóloga', 'geógrafa', \
                  'termoquímica', 'neuropsicológico', 'psicológico', 'gastrónoma', 'humanístico', 'científica-', \
                  'astronomo', 'quimico', 'psicofisiológico', 'eco-innovación', 'epidemióloga', 'alquímica', 'geóloga', 'paleontóloga', \
                  'fisiológica', 'científico-humanista', 'psicofisiológica', 'científi', 'científica-tecnológica', 'tecnocientífica', \
                  'tecnologí', 'oceanógrafo', 'psíquico', 'psico', 'paleontólogo', 'sicológica', 'bioquímico', 'mercadológico', \
                  'semiólogo', 'químico-farmacéutica', 'sicóloga', 'egiptólogo', 'geógrafo', 'fisiológico', 'primatóloga', \
                  'psicosomático', 'digitalización', 'biotecnológica', 'algólogo', 'propioceptivo', 'geoquímica', \
                  'matemático', 'tecnociencia', 'mental', 'técnico-científico', 'científicos-', 'psicológica', \
                  'psíquica', 'musculo-esquelético', 'físico-química', 'radioquímica', 'microbiológico', 'brióloga', \
                  'astrónomas', 'micóloga', 'astrofotógrafo', 'tecnológicas', 'zoóloga', 'tecnológicos', 'tecnología-', \
                  'oceanógrafa', 'antropóloga', 'agroquímica', 'físico-', 'cosmólogo', 'geoquímico', 'químico-farmacéutico', \
                  'físico-matemático', 'físico-químico', 'científico-tecnológica', 'pseudocientífica', 'tecnologización', \
                  'embriólogo', 'ecóloga', 'etóloga', 'alquímico', 'virólogo', 'histoquímica', 'sicológico', 'oncóloga', \
                  'fotoquímica', 'electroquímica', 'científico-tecnológico', 'electroquímico', 'sociológica', 'radióloga', \
                  'psicofísico', 'científico-tecnológicos', 'tísica' 'biogeoquímico', 'entomólogo', 'psico-físico',\
                  'nanotecnológico', 'aracnólogo', 'psicosomática', 'fitoquímico', 'fisiólogo', 'biogeoquímica', \
                  'mastozoólogo', 'científico-técnico', 'etólogo', 'neurocientífica', 'fisióloga', 'tecnología',\
                  'astrobiólogo', 'vulcanólogo', 'médico-científico', 'bacteriológico', 'zoólogo', 'física-', \
                  'biofísica', 'astrofísica', 'socióloga', 'físico-matemática', 'productivo', 'microbióloga', \
                  'bioquímica', 'agroquímico', 'innovación', 'etnóloga', 'estereoquímica', 'tocólogo', \
                  'ecólogo', 'científico-cultural', 'sismólogo', 'astróloga', 'malacólogo', \
                  'microbiólogo', 'ecoinnovación', 'acientífica', 'patóloga', 'fisicoquímico', \
                  'biotecnológico', 'innovativo', 'fisica', 'neuropsicológica', 'geólogo', 'algóloga', \
                  'astrofísico', 'ufólogo', 'físico-deportiva', 'fisicoquímica', 'científico-', \
                  'científico-técnica', 'briólogo', 'humanística', 'egiptóloga', 'psicofísica', \
                  'tísico', 'psico-física', 'tecnológia', 'neuroquímica', 'fotoquímico', \
                  'petroquímico', 'ufológica', 'taxónoma', 'tecnológicamente', 'neuroquímico', \
                  'astrobiólogos', 'innovativa', 'científico-académico', 'tecnocientífico', \
                  'científico-profesional', 'psico-emocional', 'físico-mental', \
                  'mitólogo', 'cognitiva', 'micólogo', 'ciencia', 'técnico-científica', 'astrónomos', \
                  'arqueólogo', 'arqueóloga', 'nano-tecnología', 'fisico']

es_art = ["arquitecto", "escultor", "pintor", "escritor", "poeta", "bailarín", "actor", "fotógrafo",
       "arquitecta", "escultora", "pintora", "escritora", "poetisa", "bailarina", "actora",
       "fotógrafa"]
es_art_expansion = [
'viñetista', 'magistrada', 'restaurador', 'co-editora', 'galerista', 'geógrafa', 'apólogo', 'traductora',
    'musicóloga', 'poesía-', 'etnógrafa', 'fotógrafos', 'ector', 'cinematógrafo', 'poetisas', 'escultista',
    'coreografió', 'editora', 'hispanista', 'arquitect', 'autor', 'exvedette', 'camarógrafo', 'ceramista',
    'orfebre', 'exboxeador', 'autora', 'detractora', 'baile', 'literato', 'apelante', 'historiógrafo',
    'músico', 'cineasta', 'ex-boxeador', 'empleadora', 'pintó', 'juzgadora', 'narrador', 'demandante',
    'sentenciadora', 'historiador', 'escultórico', 'cinefotógrafo', 'poeta-', 'poets', 'demandada',
    'bioarquitectura', 'marmolista', 'filósofa', 'bailadora', 'actor-', 'literata', 'arquitectónica',
    'comediante', 'escultórica', 'bailón', 'cantante', 'arquitectural', 'dibujante', 'coreógrafo',
    'esculpió', 'poetiza', 'arquitectura', 'fotografa', 'arquitectonica', 'cintora', 'coreógrafa',
    'fotoperiodista', 'artista', 'co-protagonista', 'danzarín', 'pinturero', 'actoral', 'astrofotógrafo',
    'interiorista', 'paleógrafo', 'libretista', 'ebanista', 'tipógrafo', 'cantautora', 'escritor-',
    'bailaría', 'poet', 'infoarquitectura', 'mitógrafo', 'ingeniero', 'infractora', 'accionante',
    'coreografo', 'arquitectónico', 'retratista', 'bailaré', 'bailador', 'arquitetura', 'bailó',
    'urbanista', 'museólogo', 'videoartista', 'trovador', 'biógrafa', 'etnógrafo', 'cartógrafo',
    'trovadora', 'bailarines', 'ilustrador', 'arquitectonico', 'compositora', 'cultora', 'abogada',
    'imputada', 'actorazo', 'cantadora', 'acuarelista', 'danza', 'codemandada', 'proyectista',
    'filósofo', 'probatoria', 'bailar', 'bailaor', 'diseñadora', 'personaje', 'actriz', 'escenógrafa',
    'coautora', 'vedette', 'constructor', 'arquitectos', 'novelista', 'bailaora', 'etnóloga',
    'camarógrafa', 'dramaturga', 'escultura', 'paisajista', 'cervantista', 'guionista', 'bailarin',
    'denunciante', 'pornógrafo', 'fotográfo', 'videógrafo', 'ensayista', 'escultores', 'querellante',
    'arquitect@s', 'historiadora', 'impugnante', 'co-autora', 'alegada', 'tallista', 'co-guionista',
    'coguionista', 'bailarinas', 'folclorista', 'filóloga', 'fotografo', 'escultoras', 'coreografiada',
    'bail', 'dramaturgo', 'coprotagonista', 'escenógrafo', 'ballet', 'bailao', 'ateopoeta', 'narradora',
    'historietista', 'danzarina', 'procuradora', 'cantautor', 'bailada', 'cantora', 'impugnada',
    'filólogo', 'cuentista', 'exbailarina', 'coescritor', 'humorista', 'poesía', 'muralista', 'litógrafo',
    'poema', 'museógrafo', 'baila', 'arquitectas', 'fotógrafas', 'arqueólogo', 'arqueóloga', 'animador',
    'reconstructora', 'ilustradora', 'bailara', 'museóloga', 'coreógrafas']

es_male = ["masculino", "hombre", "niño", "hermano", "él", "hijo", "hermano", "padre", "papá", "tío",
       "abuelo"]
es_male_exp = ['primo-hermano', 'hombre-lobo', 'mujer-hombre', 'masculinizado', 'másculino', 'sobrino', 'superhombre', 'amigo', 'padre-madre', '-hermano', 'muchacho', 'esposa', 'adulto', '-quien', 'dios-hombre', 'alguien', 'mamá-', 'nene', 'hombres–', 'masculine', 'niños–', 'hijastro', 'preadolescente', 'masculinismo', 'marido', 'diciéndolo', 'hombre-mujer', 'niño(a', 'masculina', 'femenil', 'ella', 'quien', 'hijo(s', 'abuel@s', 'cuñado', 'hombr', 'tía', 'amiguito', 'entonces', 'femenino', 'bisabuelo', 'femeninas', 'abuelito', 'bebé', 'hermanastro', 'madre', 'eadie', 'niña', 'femenina', 'papá-', 'suegra', 'herman', 'avergonzarlo', '―quien', 'ermano', 'niños/', 'papito', 'hija', 'nadie', 'masculinas', 'nieto', 'padre–', 'niños-as', 'padre-', 'suyo', 'hombrecillo', 'diciéndole', 'mamá', '-hijo', 'perro', 'quién', 'niño-', 'enfemenino', 'hermanito', 'hombre-', 'cachorro', 'progenitor', 'hombre-masa', 'ella–', 'madre-bebé', 'primo', 'masculinos', 'ellas–', 'hombre-animal', 'tatarabuelo', 'abuela', 'padrecito', 'femenina-', 'novio', 'hombre–', 'mami', 'pre-adolescente', 'bebé-', 'masculinizada', 'hermana', 'hermanos-', 'hombre-pez', 'tatarabuela', 'masculino-femenino', '-hombre', 'esposarlo', 'hermano-', 'hombres-lobo', '-padre', 'hermanita', 'tía-abuela', 'abuelete', 'femeninos', '¿alguien', 'hombrea', 'abuelo-', 'hombres/', 'papi', 'masculinidad', 'medio-hermano', 'adolescente', 'sexo-género', 'hijos–', 'aquello', 'hombre-pájaro', 'éste', 'hombretón', 'hombress', 'chico', 'hijita', 'bisabuela', 'abuelita', 'hijo-', 'unhombre', 'masculino-', 'hombre/', 'suegro', 'demonio', 'sobrino-nieto', 'femenino-', 'padrastro', 'hijito', 'álguien', 'herman@', 'tío-abuelo', 'hermanó', 'esposo', 'hombrecito']

es_female = ["femenino", "mujer", "niña", "hermana", "ella", "hija", "hermana", "madre", "mamá", "tía",
         "abuela"]
es_female_exp = ['mujer-hombre', 'ex-esposa', 'masculinizado', '-esposa', 'niño', 'muchacha', 'másculino', 'abuelo', 'nuera', 'esposa', 'abrazarla', 'padre-madre', 'niña-', 'mujer–', '-hermana', 'extrañarla', 'alguien', 'mamá-', 'madrea', 'masculine', 'hermana-', '-madre', 'mujer-', 'hijastro', 'papá', 'masculinismo', 'marido', 'femeniles', 'hombre-mujer', 'comadre', 'masculina', 'femenil', 'piamadre', 'mujerón', 'perrita', 'cuñada', 'abuel@s', 'ex-mujer', 'abuelito', 'femeninas', 'mujerona', 'hermanastro', 'sobrinita', 'amiguita', 'lujuriosamente', 'femenina', 'papá-', 'suegra', 'exmujer', 'mujerzuela', 'mujercita', 'padre', 'progenitora', 'ahijada', 'mujere', 'masculinas', 'regañarla', 'espantarla', 'diciéndola', 'niñita', 'amiga', 'aquela', 'enfemenino', 'mujeril', 'hermanito', 'muchachita', 'intimidarla', 'ella–', 'él', 'cortejarla', 'masculinos', 'abuela-', 'ellas–', 'exsuegra', 'juvenil', 'mujerhoy', 'nena', 'femenina-', '-mujer', 'novio', 'nieta', 'mami', 'chiquilla', 'madre–', 'masculinizada', 'desposarla', 'llorarla', 'niñata', 'hija-', 'madrastra', 'tatarabuela', 'masculino-femenino', 'chica', 'hijastra', 'tía-abuela', '-padre', 'hermanita', 'mujers', 'sobrina-nieta', 'femeninos', 'masculino', 'asustarla', 'mujer-madre', 'abuelo-', 'hermanastra', 'hombre(s)/mujer', 'hijo', 'hermano', '-hija', 'adolescente', 'amorosamente', 'aquello', 'supermujer', 'hijita', 'desnudarla', 'bisabuela', 'abuelita', 'sobrina', 'hijas-', 'exesposa', 'masculino-', 'jovencita', 'gatita', 'femenino-', 'felizmente', 'herman@', 'hermanó', 'esposo', 'herma', 'muje']

es_upperclass = ["Agustina", "Martina", "Josefa", "Antonia", "Sofía", "Isidora", "Cristóbal",
              "Sebastián",
              "Agustín", "Alonso", "Joaquín", "León", "Ignacio", "Julieta",
              "Matilde"]  # upper class names
es_lowerclass = ["Shirley", "Yamileth", "Sharon", "Britney", "Maryori", "Melody", "Nayareth",
              "Yaritza",
              "Byron", "Brian", "Jason", "Malcon", "Justin", "Jeremy", "Jordan", "Brayan", "Yeison",
              "Yeremi", "Bairon", "Yastin"]  # lower class/migrant names

es_good = ["caricia", "libertad", "salud", "amor", "paz", "animar", "amistad", "cielo", "lealtad",
        "placer", "diamante", "gentil", "honestidad", "suerte", "arcoiris", "diploma", "regalo",
        "honor", "milagro", "amanecer", "familia", "alegría", "felicidad", "risa", "paraíso",
        "vacación", "paz", "maravilloso", "maravillosa"]  # positive terms
es_good_exp = ['paraísos', 'convicción', 'paradisíacos', 'confraternidad', 'complacería', 'valentía', 'obsequio', 'bonita', 'misericordioso', 'paradisíacas', 'gozo', 'diamantado', 'remanso', 'tristeza', 'caricias', 'educación', 'acaricie', 'martirio', 'familia.-', 'familiar-', 'reconcilia', 'semibrillante', "d'honor", 'entretener', 'hamor', 'salubridad', 'complacerá', 'paraiso', 'homenajeada', 'suert', 'mar', 'colore', 'regalame', 'vidad', 'gentilis', 'diamantina', 'animarle', 'diplomatique', 'emoción', 'uerte', 'pubertad', 'suertes', 'fa-milia', 'amarillo', 'complacerme', 'miamor', 'doctorado', 'amatista', 'desventura', 'animarnos', 'humilde', 'milagrosa', 'lealtades', 'afortunada', 'orgasmo', 'orgullo', 'malnutrición', 'independencia', 'frugalidad', 'locura', 'pacífico', 'diamantada', 'gritito', 'cristo', 'pro-familia', 'famila', 'equisalud', 'obsequios', 'anocheceres', 'acariciadora', 'tesoro', 'animarlas', 'diplomada', 'bosteza', 'cafesalud', 'complaceré', 'diplom', 'bienestar', 'jovial', 'alegre', 'deshonor', 'honro', 'dignidad', 'pernoctación', 'sanidad', 'rigurosidad', 'milagrosos', 'posvacacional', 'amorosa', 'infortunio', 'desierto', 'vacacionales', 'jesucristo', 'vacacionar', 'colorea', 'alegres', 'regale', 'bachillerato', 'desvanecerse', 'alegr', 'satisfacción', 'divertir', 'auto-regalo', 'paisaje', 'regalarlo', 'sentimentalidad', 'acaricié', 'generosidad', 'fortuna', 'honran', 'luna', 'regalón', 'probidad', 'milagrito', 'reconciliado', 'carcajea', 'ingenuidad', 'invitarles', 'desgracia', 'bronca', 'rabia', 'caballeroso', 'familar', 'amistades', 'vivaz', 'vacaciones', 'oasis', 'maravillé', 'acariciarla', 'reine', 'entusiasmo', 'subfamilia', 'amoralidad', 'deleite', 'acaricio', 'colori', 'plateado', 'regaláis', 'maravillosos', 'maraville', 'semanaamanecer', 'respeto', 'honorifico', 'animarlo', 'residencia', 'semilibertad', 'reconciliador', 'ciel', 'encantador', 'mistad', 'unicolor', 'gentile', 'paradisíaca', 'acariciandole', 'camaradería', 'amorosidad', 'diplomacy', 'acariciándola', 'diplomó', 'familias-', 'placentera', 'maravillar', 'paraisos', 'coherencia', 'ecuanimidad', 'diamantino', 't13.cl', 'inspirar', 'cordial', 'colorean', 'milagrosamente', 'homenaje', 'acariciarle', 'despertarte', 'gracia', 'maravillan', 'bertad', 'risa-', 'complacernos', 'acariciar', 'sentimento', 'obediencia', 'compasión', 'liberalidad', 'familiar', 'libertas', 'irises', 'amoral', 'infierno', 'paradisíaco', 'afable', 'unicornio', 'estancia', 'tranquilidad', 'animarte', 'esperarlo', 'honrara', 'homenajeado', 'locación', 'maravillarte', 'honorífica', 'amori', 'alegría', 'gustazo', 'resplandecer', 'amistas', 'gante', 'rayo', 'maravil', 'lágrima', 'grandioso', 'bachiller', 'objetividad', 'alegrìa', 'liberté', 'semi-libertad', 'fraternidad', 'libertades', 'humildad', 'magnífica', 'pacífica', 'gentita', 'coloree', 'reconciliación', 'santo', 'regalar', 'cielorraso', 'perforador', 'libertarse', '-amor', 'despertarme', 'animara', 'regalarselo', 'monocristal', 'grandiosa', 'bellísima', 'complacerlas', 'arco-iris', 'invernación', 'regalazo', 'azules', 'despertarnos', 'amis', 'visitación', 'desconfianza', 'homenajeo', 'cristalizador', 'milagros', 'libertaria', 'respetabilidad', 'familiarízate', 'desamor', 'fraternal', 'indepencia', 'reconciliacion', 'subnutrición', 'coloreaba', 'inquebrantable', 'fantástico', 'deseo', 'placet', 'diplomarse', 'honra', 'hermosa', 'homenajeó', 'anillo', 'diplomado', 'acariciada', 'alegren', 'invitar', 'rectitud', 'vacacionista', 'rabieta', 'noviazgo', 'suer', 'vitalidad', 'regalonear', 'complacerlos', 'regal', 'estupenda', 'hermoso', 'amor/', 'cielo-', 'animaros', 'euforia', 'paradisiaca', 'familia–', 'diplomatura', 'fortuna-', 'familiaris', 'arcoíris', 'preciosa', 'fabulosa', 'brillante', 'júbilo', 'acariciandolo', 'seriedad', 'enemistad', 'iris', 'fabuloso', 'amabilísimo', 'animarlos', 'regocijo', 'gentileza', 'tregua', 'beca', 'despertar', 'sonrisa', 'afortunadamente', 'amanece', 'atardece', 'serenidad', 'prosperidad', 'azul', 'amable', 'desgraciao', 'lafamilia', 'maravillado', 'gusto', 'helicidad', 'libertado', 'caballerosidad', 'placer-', 'bondadoso', 'coloreen', 'motivarles', 'acariciandola', 'sonrrisa', 'libertario', 'diamantes', 'colgante', 'sonrisilla', 'amanecería', 'cariño', 'amistosa', 'amaneces', 'perpetuidad', 'risas-', 'cariño-', 'carcajadas', 'familial', 'emocionalidad', 'complacer', 'carcajada', 'confianza', 'misericordiosa', 'pasión', 'honorado', 'sinceridad', 'convicci', 'libertadora', 'magnifico', 'regalos', 'emoci', 'alegria', 'horizonte', 'noble', 'acariciaba', 'familiarmente', 'alegrón', 'anochece', 'suerte-', 'minsalud', 'vacacionistas', 'honores', 'calvario', 'animaran', 'resplandor', 'médico-sanitaria', 'honorum', 'sonrisas', 'sonriéndole', 'insatisfacción', 'insinceridad', 'animarse', 'amaneci', 'circonio', 'inmortalidad', 'perseverancia', 'salud-enfermedad', 'gentilidad', 'nutrición', 'moralidad', 'alquilación', 'animarla', 'morio', 'suertudos', 'honorífico', 'infelicidad', 'complacerla', 'ilusión', 'displacer', 'liberta', 'dios', 'guirnalda', 'paraje', 'colorinche', 'sonrisita', 'sol', 'gielo', 'ocaso', 'traición', 'concordia', 'ensueño', 'amistó', 'amista', 'justicia', 'nubes', 'sacratísimo', 'suertudo', 'libertad-', 'estupendo', 'amis-', 'tristeza-', 'familia-escuela', 'honrar', 'animarles', 'vacacional', 'legitimidad', 'rinconcito', 'edén', 'perforante', 'insalud', 'e-salud', 'ilusionar', 'reconciliada', 'cristal', 'encantadísima', 'profesionalidad', 'risas', 'complacerles', 'bellísimo', 'zielo', 'iridiscente', 'trueno', 'maravillo', 'coloreó', 'karicia', 'esperanza', 'honradez', 'regalarle', 'familia-', 'amaneceres', 'encantadora', 'cortés', 'prevención', 'paradisiaco', 'honor-', 'postgraduado', 'magnífico', 'amore', 'maravillos', 'suertudas', 'gentis', 'famili', 'libertar', 'motivar', 'profamilia', 'acariciante', 'pueblito', 'post-vacacional', 'honró', 'amoroso', 'maravillada', 'talud', 'amorío', 'alentar', 'regalito', 'paradisiacas', 'atardecer', 'diplomat', 'soberanía', 'piadoso', 'cariñosa', 'ocupación', 'escrupulosidad', 'ecosalud', 'libertada', 'turistificación', 'olvidad', 'honorar', 'placentero', 'estivación', 'amilia', 'placeres', 'complacerte', 'corazón', 'postítulo', 'bondadosa', 'sobrefamilia', 'honorabilidad', 'familía', 'prodigio', 'coloreo', 'regalona', 'acaricia', 'regaléis', 'crepúsculo', 'regalare', 'reconciliadora', 'circonita', 'animare', 'complacerlo', 'increíble', 'santísimo', 'honour', 'regalarla', 'paradisiacos', 'erotismo', 'cielos', 'milagrero', 'despertarse', 'homenajea', 'tiniebla', 'fortunei', 'diplomas', 'entusiasmar', 'cariñoso', "d'amor", 'censalud', 'sentimental', 'familiars', 'eduación', 'complacerle', 'pareja', 'armonía', 'porquería', 'obsequioso', 'homenajee', 'celestial', 'anochecer', 'broma', 'resucito', 'milagroso', 'autosatisfacción', 'regalé', 'cohabitación', 'acariciara', 'deshonestidad', 'premio', 'hermandad', 'sacrilegio', 'refugio', 'postvacacional', 'salud-', 'sonrisa.-', 'amor-', 'gratitud', 'espiritualidad', 'acariciándole', 'grima', 'divertirle', 'compañerismo', 'ilegitimidad', 'scielo', 'floración', 'máster', 'fantástica', 'essalud', '-hola', 'aielo', 'fidelidad', 'hermosísima', 'milagrosas', 'suertuda', 'diamant', 'manto', 'milagrera', 'famil', 'leal', 'unicornios', 'ternura', 'sonrisota', 'intranquilidad', 'hermosísimo', 'democracia', 'simpatía', 'deslealtad', 'multicolor', 'regalaré', 'alegrías', 'bendito', 'maravilla', 'amar', 'colorin']

es_bad = ["abuso", "choque", "suciedad", "asesinato", "enfermedad", "accidente", "muerte",
       "sufrimiento", "veneno", "hedor", "apestar", "ataque", "asalto", "desastre", "odio",
       "contaminación", "tragedia", "divorcio", "cárcel", "pobreza", "fea", "feo", "cáncer",
       "matar", "vómito", "bomba", "maldad", "podrido", "podrida", "agonía", "terrible",
       "horrible", "guerra", "repugnante"]
es_bad_exp = ['malote', 'reclusas', 'siniestro', 'accidentado', 'angustia', 'emboscada', 'bombs', 'corrible', 'aberrante', 'defensa-ataque', 'sanguinolenta', 'repulsivo', 'uerra', 'crimen', 'tristeza', 'chulo', 'malevolencia', 'indigesto', 'guarecer', 'catástrofe', 'guardiacárcel', 'embriagador', 'hundimiento', 'carcel', 'enveneno', 'guerracivilista', 'muriera', 'acidificación', 'atacador', 'temor', 'bombay', 'avaricia', 'asesinarlo', 'menuda', 'càncer', 'divorcia', 'sinrazón', 'horrenda', 'sufrimientos', 'deleznable', 'falleciera', 'inmunodepresión', 'com.-estar', 'conmoción', 'trágica', 'asqueroso', 'acidente', 'vasoespasmo', 'ponzoña', 'violencia', 'indignante', 'estornudo', 'desigualdades', 'saxitoxina', 'congoja', 'intentado', 'náuseas', 'desventura', 'deshecho', 'húmedad', 'abrasividad', 'empecinado', 'venenosas', 'torturar', 'antiguerra', 'muert', 'desastrado', 'bombardeo', 'desaparici', 'bien-estar', 'malo', 'emboscado', 'pobrezas', 'sobreexplotación', 'podrirse', 'insensatez', 'estarnos', 'tufo', 'buso', 'locura', 'motobomba', 'soberbia', 'decido', 'horrorosa', 'olorcito', 'absorbencia', 'turbobombas', 'cohonestar', 'masacre', 'enfrentamiento', 'asaltó', 'espantosa', 'envenenarle', 'contrataque', 'afección', 'ciberataques', 'matrimonio', 'prisión-', 'execrable', 'guay', 'ataqué', 'enferm', 'humo', 'malditismo', 'adenocarcinoma', 'molestarle', 'asesinarme', 'resentimiento', 'intolerable', 'anti-guerra', 'dolencia', 'tonto', 'flaca', 'racismo', 'choques', 'divorcios', 'doloroso', 'abusé', 'efermedad', 'matarlas', 'desagradable', 'prisin', 'hepatocarcinoma', 'golpe', 'chiquillada', 'ignorante', 'asaltante', 'eutrofización', 'enfermeda', 'asesinarla', 'cã¡rcel', 'rabia', 'cáncer', 'asesine', 'empobrezca', 'desastroso', 'desastres', 'ignorancia', 'cabreada', 'semejante', 'padecimiento', 'odios', 'marginación', 'carcinoma', 'desazón', 'violaciónes', 'aroma', 'frialdad', 'postguerra', 'divorciado', 'jodida', 'desvalimiento', 'desigualdad', 'inequidad', 'contaminaciones', 'espantoso', 'colisionador', 'vomito', 'escalofriante', 'agonías', 'hediondez', 'carcomida', 'guerra-', 'andarse', 'cancer', 'callosidad', 'hediondo', 'vileza', 'inmunodeficiencia', 'podrir', 'insultante', 'jodido', 'asesinatos', 'pegajosidad', 'asesinaría', 'tentado', 'podri', 'envenenador', 'desastrosamente', 'dolorosa', 'putrefacta', 'pesadilla', 'cruel', 'corrosión', 'cabronada', 'disentimiento', 'corrosividad', 'venenoso', 'divorce', 'dispararle', 'vaciedad', 'podredumbres', 'polución', 'aterrorizante', 'invasión', 'bajito', 'cabron', 'asalte', 'anticontaminación', 'deceso', 'triste', 'tontita', 'muerte–', 'reblandecido', 'lubricidad', 'atacarle', 'xenofobia', 'raro', 'prepotente', 'matarse', 'divorcié', 'desesperación', 'accidental', 'sobre-explotación', 'antirracismo', 'analfabetismo', 'abus', 'fetidez', 'acontecida', 'estupidez', 'asalta', 'horroroso', 'guerra–', 'desocupación', 'angustio', 'atraco', 'viejito', 'bombeó', 'entreguerra', 'torturarle', 'muertes', 'andarle', 'empelado', 'insufrible', 'rencor', 'electrochoque', 'extraño', 'abusas', 'podris', 'penuria', 'nauseabundos', 'asesinará', 'ajuriaguerra', 'secuestro', 'muerto', 'terremoto', 'terriblemente', 'bestial', 'mareos', 'insecticida', 'venenos', 'diarrea', 'desprecio', 'encarcela', 'empeñado', 'salinización', 'apitoxina', 'homicidio', 'homofobia', 'fermedad', 'tragicomedia', 'divorcie', 'envenena', 'asesiné', 'cataclismo', 'bomb', 'rugosidad', 'gordita', 'exreclusos', 'basalto', 'abuse', 'suciedades', 'limpidez', 'codicia', 'abusivo', 'suicidio', 'descarrilamiento', 'preguerra', 'fracaso', 'violaciã³n', 'patología', 'baboso', 'cristianofobia', 'derrumbe', 'cárceles', 'disparo', 'anti-cáncer', 'accidentarse', 'esguerra', 'engreída', 'nauseabundo', 'mucosidad', 'desaseo', 'emboscó', 'divorciarse', 'contaminaciã³n', 'reclusa', 'sobrexplotación', 'angustioso', 'parachoque', 'accidents', 'enojo', 'acostumbrado', 'pudica', 'maltratos', 'tonta', 'violación', 'desembarco', 'vinciguerra', 'desastrozo', 'colisión', 'olor', 'loquita', 'maltrato', 'tuberculosis', 'remordimiento', 'podridas', 'desesperante', 'fallecer', 'matanza', 'asesinó', 'fétido', 'cabrón', 'empequeñecimiento', 'asesinarlos', 'hollín', 'pudría', 'antiveneno', 'ira', 'cárcel-', 'podido', 'contragolpe', 'sobreviviente', 'envenenó', 'guapo', 'orrible', 'humedad', 'mareo', 'horripilante', 'sentimiento', 'delito', 'deicidio', 'anti-humedad', 'asaltar', 'probreza', 'anticáncer', 'desastroza', 'ataquen', 'sufriente', 'podredumbre', 'contra-ataque', 'pestilencia', 'muerta', 'cortocircuito', 'choquen', 'estarlos', 'angustió', 'miserable', 'gorda', 'incidente', 'oleosidad', 'aminación', 'fealdad', 'detona', 'abusos', 'chulito', 'crueldad', 'filicidio', 'fanatismo', 'desaturación', 'horrendo', 'demencia', 'machismo', 'enferme-', 'fallecimiento', 'indigencia', 'embrutecimiento', 'catástrofes', 'asesinado', 'infamante', 'divorciaría', 'bombona', 'nauseabunda', 'extraña', 'impureza', 'oloroso', 'bombe', 'bondad', 'desacostumbrado', 'atacante', 'gordito', 'mordedura', 'asqueada', 'contaminante', 'venenosos', 'prisi', 'melanoma', 'matarle', 'asesinase', 'tragedias', 'estarles', 'molestar', 'extrañamiento', 'catastrófico', 'asquerosa', 'asesinarte', 'melancolía', 'tiroteo', 'desaparición', 'barbarie', 'impotencia', 'combate', 'ncer', 'sucio', 'hipocresía', 'excarcelada', 'post-guerra', 'patito', 'vómitos', 'matrimonió', 'divorcian', 'maldita', 'despreciable', 'matrimonio-', 'aburrimiento', 'escorpión', 'egoísmo', 'devastador', 'accident', 'horriblemente', 'desertificación', 'angustiosa', 'pobreza-', 'multihomicidio', 'guapa', 'matrimonia', 'turbina', 'irremediable', 'lamuerte', 'cobardía', 'emperrado', 'presidio', 'asesinar', 'hecatombe', 'choqué', 'vanidad', 'chocante', 'espanto', 'abusado', 'cã¡ncer', 'trágico', 'biodegradación', 'embate', 'enfermes', 'injusticia', 'escandaloso', 'salud-enfermedad', 'accidenta', 'accidente-', 'violaci', 'atroz', 'pre-guerra', 'degradación', 'guerreó', 'guerras', 'devastadora', 'desasimiento', 'culera', 'tumore', 'emfermedad', 'antichoque', 'vergonzante', 'sanguijuela', 'matarme', 'fraude', 'bombita', 'traición', 'marginalidad', 'espeluznante', 'chiquita', 'atropellamiento', 'reclusos', 'tragedies', 'asaltos', 'estarme', 'matarles', 'cabrona', 'humillante', 'desangramiento', 'cidente', 'degradante', 'cánceres', 'ataques', 'envenenada', 'prejuicio', 'matrimonial', 'cabro', 'atontar', 'matarla', 'percance', 'malinchismo', 'laenfermedad', 'carceleta', 'tontito', 'maldecido', 'matrimoniali', 'estarle', 'espasmo', 'subachoque', 'abominable', 'miedo', 'ofensiva', 'recluso', 'ciberataque', 'desastrada', 'desnutrición', 'flaco', 'siniestrado', 'malicia', 'muer', 'tiroteó', 'pestilente', 'enfermedades', 'duelo', 'espante', 'leucemia', 'insolente', 'olorcillo', 'desaminación', 'broncoespasmo', 'tipa', 'empobrecimiento', 'hedionda', 'divorció', 'accidentalista', 'vomita', 'encontronazo', 'accidentalidad', 'atacarlo', 'laguerra', 'bombas', 'reblandecida', 'asesinara', 'precariedad', 'indecible', 'antihumedad', 'indiscriminado', 'muerto-', 'maldecida', 'explosion', 'vida-muerte', 'pobres', 'soso', 'antídoto', 'vaho', 'accidentes', 'contraataques', 'divorcien', 'preso', 'coche-bomba', 'turbobomba', 'putrefacto', 'asedio', 'vengarse', 'divorciaran', 'alma_podrida', 'amargura', 'derrumbamiento', 'accidentó', 'disimularse', 'autobomba', 'enfermedad-', 'infestar', 'abusador', 'detonada', 'catastróficamente', 'divorciara', 'otosclerosis', 'bajita', 'inseguridad', 'victimismo', 'encarcelado', 'asesinarle', 'desesperanza', 'parásito', 'ensordecido', 'prisión', 'electrobombas', 'vomitar', 'estarlo', 'asesino', 'muerte-', 'indecente', 'estomacal', 'nausea', 'pesadumbre', 'cazarle', 'abatimiento', 'atentado', 'divorciada', 'cochinada', 'maloliente', 'estar', 'abusones', 'golpetazo', 'colapso', 'asesinada', 'soledad', 'putrefacción', 'podridos', 'bombarda', 'partidazo', 'intrusismo', 'náusea', 'vaciada', 'cáncer-', 'contaminaci', 'angustiaba', 'prision', 'contraataque', 'abusó', 'divorciándose', 'desgarramiento', 'arrepentimiento', 'sintomatología', 'miseria', 'asaltara', 'impurezas', 'matarlo', 'matarte', 'divorciar', 'maldiciente', 'envenene', 'pos-guerra', 'descontaminación', 'molestarse', 'tumor', 'deforestación', 'nauseas', 'desvanecimiento', 'revanchismo', 'estúpido', 'accidentalmente', 'repudiable', 'feísima', 'desastrosa', 'magnicidio', 'contaminantes', 'encarcelada', 'cólico', 'discrimen', 'mezquindad', 'odioso', 'letal', 'infección', 'taque', 'pútrido', 'detestar', 'porosidad', 'posguerra', 'empecinada', 'explosivo', 'acoso', 'chunga', 'infertilidad', 'descalabro', 'desempleo', 'avergonzante', 'arrogancia', 'accidentada', 'envenenarlo', 'venenosa', 'sanguinolento', 'podrían', 'estarte', 'asesina', 'putona', 'atestar', 'muertos-', 'efluvio', 'catastrofe', 'impactación', 'matrimoniale', 'estómago', 'toxina', 'electrobomba', 'gordo', 'inmoralidad', 'matarlos']


### Bolukbasi words
all_bolukbasi = ["he","his","her","she","him","man","women","men","woman","spokesman","wife",
                 "himself","son","mother","father","chairman","daughter","husband","guy","girls",
                 "girl","boy","boys","brother","spokeswoman","female","sister","male","herself",
                 "brothers","dad","actress","mom","sons","girlfriend","daughters","lady",
                 "boyfriend","sisters","mothers","king","businessman","grandmother","grandfather",
                 "deer","ladies","uncle","males","congressman","grandson","bull","queen","businessmen",
                 "wives","widow","nephew","bride","females","aunt","prostatecancer","lesbian","chairwoman",
                 "fathers","moms","maiden","granddaughter","youngerbrother","lads","lion","gentleman",
                 "fraternity","bachelor","niece","bulls","husbands","prince","colt","salesman","hers",
                 "dude","beard","filly","princess","lesbians","councilman","actresses","gentlemen",
                 "stepfather","monks","exgirlfriend","lad","sperm","testosterone","nephews","maid",
                 "daddy","mare","fiance","fiancee","kings","dads","waitress","maternal","heroine",
                 "nieces","girlfriends","sir","stud","mistress","lions","estrangedwife","womb",
                 "grandma","maternity","estrogen","exboyfriend","widows","gelding","diva",
                 "teenagegirls","nuns","czar","ovariancancer","countrymen","teenagegirl","penis",
                 "bloke","nun","brides","housewife","spokesmen","suitors","menopause","monastery",
                 "motherhood","brethren","stepmother","prostate","hostess","twinbrother",
                 "schoolboy","brotherhood","fillies","stepson","congresswoman","uncles","witch",
                 "monk","viagra","paternity","suitor","sorority","macho","businesswoman",
                 "eldestson","gal","statesman","schoolgirl","fathered","goddess","hubby",
                 "stepdaughter","blokes","dudes","strongman","uterus","grandsons","studs","mama",
                 "godfather","hens","hen","mommy","estrangedhusband","elderbrother","boyhood",
                 "baritone","grandmothers","grandpa","boyfriends","feminism","countryman","stallion",
                 "heiress","queens","witches","aunts","semen","fella","granddaughters","chap",
                 "widower","salesmen","convent","vagina","beau","beards","handyman","twinsister",
                 "maids","gals","housewives","horsemen","obstetrics","fatherhood","councilwoman",
                 "princes","matriarch","colts","ma","fraternities","pa","fellas","councilmen",
                 "dowry","barbershop","fraternal","ballerina"]

male_bolukbasi = ["he","his","him","man","men","himself","son","father","husband","guy",
                 "boy","boys","brother","male","brothers","dad","sons","boyfriend","grandfather",
                 "uncle","males","grandson","nephew", "fathers","lads","gentleman",
                 "fraternity","bachelor","husbands","dude","beard","gentlemen","stepfather","lad",
                 "testosterone","nephews","daddy","dads","sir","bloke", "schoolboy","stepson","uncles",
                 "hubby","blokes","dudes","grandsons","godfather","boyhood",
                 "grandpa","boyfriends","fella","chap","beards","pa","fellas","fraternal"]

female_bolukbasi = ["he","his","her","she","him","man","women","men","woman","spokesman","wife",
                 "himself","son","mother","father","chairman","daughter","husband","guy","girls",
                 "girl","boy","boys","brother","spokeswoman","female","sister","male","herself",
                 "brothers","dad","actress","mom","sons","girlfriend","daughters","lady",
                 "boyfriend","sisters","mothers","king","businessman","grandmother","grandfather",
                 "deer","ladies","uncle","males","congressman","grandson","bull","queen","businessmen",
                 "wives","widow","nephew","bride","females","aunt","prostatecancer","lesbian","chairwoman",
                 "fathers","moms","maiden","granddaughter","youngerbrother","lads","lion","gentleman",
                 "fraternity","bachelor","niece","bulls","husbands","prince","colt","salesman","hers",
                 "dude","beard","filly","princess","lesbians","councilman","actresses","gentlemen",
                 "stepfather","monks","exgirlfriend","lad","sperm","testosterone","nephews","maid",
                 "daddy","mare","fiance","fiancee","kings","dads","waitress","maternal","heroine",
                 "nieces","girlfriends","sir","stud","mistress","lions","estrangedwife","womb",
                 "grandma","maternity","estrogen","exboyfriend","widows","gelding","diva",
                 "teenagegirls","nuns","czar","ovariancancer","countrymen","teenagegirl","penis",
                 "bloke","nun","brides","housewife","spokesmen","suitors","menopause","monastery",
                 "motherhood","brethren","stepmother","prostate","hostess","twinbrother",
                 "schoolboy","brotherhood","fillies","stepson","congresswoman","uncles","witch",
                 "monk","viagra","paternity","suitor","sorority","macho","businesswoman",
                 "eldestson","gal","statesman","schoolgirl","fathered","goddess","hubby",
                 "stepdaughter","blokes","dudes","strongman","uterus","grandsons","studs","mama",
                 "godfather","hens","hen","mommy","estrangedhusband","elderbrother","boyhood",
                 "baritone","grandmothers","grandpa","boyfriends","feminism","countryman","stallion",
                 "heiress","queens","witches","aunts","semen","fella","granddaughters","chap",
                 "widower","salesmen","convent","vagina","beau","beards","handyman","twinsister",
                 "maids","gals","housewives","horsemen","obstetrics","fatherhood","councilwoman",
                 "princes","matriarch","colts","ma","fraternities","pa","fellas","councilmen",
                 "dowry","barbershop","fraternal","ballerina"]

weat_words = load_weat()

def weat_3(lower=True):
    targets_1 = copy(weat_words['european_american_names_5'])
    for w in ['Jonathan', 'Stephen', 'Megan', 'Harry', 'Colleen', 'Courtney',
              'Josh', 'Heather', 'Matthew', 'Betsy', 'Katie', 'Brad']:
        targets_1.remove(w)
    targets_2 = copy(weat_words['african_american_names_5'])
    for w in ['Shaniqua', 'Tanisha', 'Malika', 'Latoya', 'Nichelle', 'Wardell',
              'Latisha', 'Shereen', 'Alphonse', 'Lakisha', 'Lavon',
              'Marcellus']:
        targets_2.remove(w)
    return Query([list(map(lambda w: w.lower() if lower else w, targets_1)),
                  list(map(lambda w: w.lower() if lower else w, targets_2))],
                 [weat_words['pleasant_5'], weat_words['unpleasant_5b']],
                 ['european american names', 'african american names'],
                 ['pleasant', 'unpleasant'])


def weat_4(lower=True):
    targets_1 = copy(weat_words['european_american_names_7'])
    for w in ['Matthew', 'Geoffrey', 'Brad', 'Neil', 'Jill', 'Todd']:
        targets_1.remove(w)
    targets_2 = copy(weat_words['african_american_names_7'])
    for w in ['Tamika', 'Latoya', 'Tanisha', 'Latonya', 'Tremayne', 'Lakisha']:
        targets_2.remove(w)
    return Query([list(map(lambda w: w.lower() if lower else w, targets_1)),
                  list(map(lambda w: w.lower() if lower else w, targets_2))],
                 [weat_words['pleasant_5'], weat_words['unpleasant_5b']],
                 ['european american names', 'african american names'],
                 ['pleasant', 'unpleasant'])


def weat_5(lower=True):
    targets_1 = copy(weat_words['european_american_names_7'])
    for w in ['Matthew', 'Geoffrey', 'Brad', 'Neil', 'Jill', 'Todd']:
        targets_1.remove(w)
    targets_2 = copy(weat_words['african_american_names_7'])
    for w in ['Tamika', 'Latoya', 'Tanisha', 'Latonya', 'Tremayne', 'Lakisha']:
        targets_2.remove(w)
    return Query([list(map(lambda w: w.lower() if lower else w, targets_1)),
                  list(map(lambda w: w.lower() if lower else w, targets_2))],
                 [weat_words['pleasant_9'], weat_words['unpleasant_9']],
                 ['european american names', 'african american names'],
                 ['pleasant', 'unpleasant'])


def weat_6(lower=True):
    # Note I swapped this around from the original so gender is always attributes
    targets_1 = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
    targets_2 = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]
    attributes_1 = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
    attributes_2 = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']
    return Query(target_sets=[targets_1, targets_2],
                 attribute_sets=[attributes_1, attributes_2],
                 target_sets_names=['career', 'family'],
                 attribute_sets_names=['male', 'female'])


def weat_7(lower=True):
    return Query([weat_words['math'], weat_words['arts']],
                 [weat_words['male_terms'], weat_words['female_terms']],
                 ['math', 'arts'], ['male', 'female'])


def weat_7_twitter(lower=True):
    targets_1 = copy(weat_words['math'])
    targets_1.remove('computation')
    targets_2 = copy(weat_words['arts'])
    targets_2.remove('symphony')
    return Query([targets_1, targets_2],
                 [weat_words['male_terms'], weat_words['female_terms']],
                 ['math', 'arts'], ['male', 'female'])


def weat_8(lower=True):
    return Query([list(map(lambda w: w.lower() if lower else w, weat_words['science'])),
                  list(map(lambda w: w.lower() if lower else w, weat_words['arts_2']))],
                 [weat_words['male_terms_2'], weat_words['female_terms_2']],
                 ['science', 'arts'], ['male', 'female'])


def weat_gender(lower=True):
    w6 = weat_6(lower)
    w7 = weat_7(lower)
    w8 = weat_8(lower)
    return Query([list(set(w6.target_sets[i]) | set(w7.target_sets[i])
                       | set(w8.target_sets[i])) for i in range(2)],
                 [list(set(w6.attribute_sets[i]) | set(w7.attribute_sets[i])
                       | set(w8.attribute_sets[i])) for i in range(2)],
                 ['career, math, and science', 'family and arts'],
                 ['male', 'female'])


def weat_gender_twitter(lower=True):
    # union of weat_6, weat_7_twitter, and weat_8
    q = weat_gender(lower)
    q.target_sets[0].remove('computation')
    # There's no need to remove 'symphony' because it is included in weat_8.
    return q


def lower_wordlist(wordlist):
    return list(set(map(lambda w: w.lower(), wordlist)))


def weat_gender_exp(lower=True):
    targets_1 = ['NASA', 'computation', 'corporation', 'Einstein', 'addition', 'calculus', 'astronomy', 'business', 'numbers', 'math', 'office', 'experiment', 'executive', 'salary', 'professional', 'geometry', 'algebra', 'science', 'technology', 'physics', 'career', 'equations', 'chemistry', 'management', 'office-', 'Biophysics', 'offices', 'additional', 'experimenters', 'geosciences', 'salariés', 'salaryman', 'NASA.gov', 'interpolations', 'experimentations', 'imputation', 'Nanotechnologies', 'Paraprofessional', 'Einsteins', 'Additionally', 'professionalise', 'managment', 'photochemistry', 'JPL', 'Showbusiness', 'experiments', 'Risikomanagement', 'businesslike', 'geometries', 'micromanagement', 'astronomic', 'cycloaddition', 'business-', 'geometrics', 'permutations', 'radiochemistry', 'executively', 'careers', 'executives', 'phytochemistry', 'corporatization', 'director', 'sciences', 'additionally', 'science-', 'computationally', 'Additionaly', 'geometric', 'experimentellen', 'algebraic', 'professiona', 'parameterizations', 'interprofessional', 'C*-algebra', 'Astronomy', 'subalgebras', 'careerism', 'biochemistry', 'office/', 'Incorporation', 'mathematik', 'Mismanagement', "Ke'ra", 'salaries', 'gastronomy', 'numbe', 'psychophysics', 'backoffice', 'computational', 'experimenter', 'Nasa', 'Freudenstein', 'Corporations', 'Levinstein', 'wavenumbers', 'experimenting', 'professionaly', 'geophysics', 'megacorporations', 'mismanagement', 'quantization', 'stereochemistry', 'Bancorporation', 'career-', 'arithmetic', 'megacorporation', 'thermochemistry', 'computerisation', 'astronomique', 'Phytochemistry', 'Aménagement', 'directorship', 'Astronomia', 'professionalising', 'Incorporations', 'SpaceX.', 'experimentally', 'semiprofessional', 'showbusiness', 'Salary', 'Precalculus', 'Microtechnology', 'Stereochemistry', 'Equations', '-algebras', 'Spacecraft', 'paraprofessional', 'mathcad', 'midcareer', 'coalgebra', 'Executives', 'geometrical', 'management-', 'salarié', 'technologique', 'Grinstein', 'corporatisation', 'fractions', 'Calculus', 'technological', 'Boxoffice', 'precalculus', 'experimented', 'boxoffice', 'technology-', 'nanotechnology', 'agribusiness', 'biophysics', 'trigonometry', 'office--', 'libreoffice', 'ellipsometry', 'geoscience', 'equational', 'Imputation', 'codirector', 'eigenfunctions', 'Addition', 'archaeoastronomy', '2867', 'MyCareer', 'astronomers', 'ofbusiness', 'C*-algebras', 'computerization', 'Geochemistry', 'corporative', 'career--', 'hypergeometric', 'additionals', 'subalgebra', 'officejet', 'maths', 'Micromanagement', 'Gastronomy', 'Executive', 'NASAA', 'careerist', 'Pataphysics', 'isometry', 'sciency', 'astrophysics', 'professionalized', 'ESA', 'technologie', 'businesses', 'technoscience', 'corporations', 'computations', 'salary.com', 'technologies', 'businesss', 'experimentation', 'Additions', 'algebras', 'multiphysics', 'managements', 'number(s', 'technologic', 'consultant', '-algebra', 'Einsteinian', 'bioscience', 'experimental', 'astronomical', 'mathematics', 'directors', 'salaried', 'sciencey', 'additionality', 'business--', 'Astronaut', 'calculi', 'Cuemath', 'geometria', 'Managements', 'salarymen', 'SpaceX', 'microphysics', 'geochemistry', 'additions', 'heliophysics', 'superalgebra', 'calcul', 'Equation', 'Bekenstein', 'equation', 'numbers--', 'postoffice', 'homework', 'geometrid']
    targets_2 = ['drama', 'marriage', 'family', 'symphony', 'cousins', 'dance', 'poetry', 'home', 'children', 'parents', 'Shakespeare', 'wedding', 'literature', 'sculpture', 'art', 'novel', 'relatives', 'sculpturing', 'nephews', 'homesite', 'dancesport', 'dancey', 'novellas', 'painting', 'WeddingWire', 'artist', 'family--', 'comedy', 'artistas', 'philological', 'familie', 'Grandparents', 'poema', 'historiography', 'literary', 'parents--', 'marriage--', 'Hamlet', 'orchestral', 'dancy', 'weddings', 'wedding--', 'familys', 'familyâ\x80\x99s', 'shakespeare', 'concerto', 'poets', 'symphonique', 'novella', 'family-', 'children-', 'arts', 'Remarriage', 'historiographical', 'children--', 'dancers', 'Beethoven', 'Shakespear', 'childrens', 'Shakespearean', 'Shakspeare', 'symphonic', 'symphonies', 'stepparents', 'Sculpture', 'literatures', 'relative', 'Melodrama', 'dance-', 'remarriage', 'Shakespears', 'yourfamily', 'dramas', 'Webnovel', 'poetics', 'poetas', 'sisters', 'home-', 'grandparents', 'children&rsquo', 'brothers', 'symphonie', 'Sculptures', 'Macbeth', 'docudrama', 'sons', 'marriages', 'Kdrama', 'brother-', 'grandsons', 'Shakespearian', 'famil', 'artstyle', 'poems', 'Docudrama', 'Literatures', 'artist(s', 'novela', 'kdrama', 'Relatives', 'novels', 'dances', 'poetic', 'artworks', 'birthparents', 'artisti', 'novelas', 'marriageable', 'stepfamily', 'bridal', 'cousin', 'novelette', "children's", 'married-', 'Symphony', 'children`s', 'mothers', 'dancing', "family'll", 'children´s', 'danced', 'childre', 'sculptural', 'literaturii', 'dramedy', 'home--', 'parents/', 'siblings', 'Literature', 'remarriages', 'poetica', 'sculpturally', 'sculptures', 'ethnomusicological', 'weddin', 'orchestra', 'sculptured', 'danceable', 'uncles', 'sculpts', 'artwork', 'dancehalls', 'poeta', 'literatury', 'grandnephews', 'Sculptural', 'artistry', 'home/', 'Wedding', 'artistic', 'Symphonies', 'literatura', 'Shakespeares', 'parents-', 'Monodrama', 'marriageCredit', 'childrenâ€', 'Shome', '\\relative', 'weddingCredit', 'Floetry']
    attributes_1 = ['grandfather', 'son', 'father', 'uncle', 'brother', 'his', 'man', 'male', 'boy', 'him', 'he', 'brother-', 'His', 'brotherinlaw', 'himself-', "father'll", 'himselfe', 'stepbrother', 'Cutfather', 'stepfather', 'himself', 'himself--', 'males', 'Kheir', 'granduncle', 'grandson', 'Grandfather', 'Bairnsfather', 'male-', 'herfather', 'nephew', 'father-', 'grandfatherly', 'Allfather', 'grandfathers', 'guy', 'brothers', 'brother--', 'He', 'Oncle', 'nuncle', 'grandnephew', 'Stepbrother', 'Hogfather', 'Dogfather']
    attributes_2 = ['mother', 'woman', 'sister', 'hers', 'she', 'aunt', 'her', 'female', 'daughter', 'grandmother', 'girl', 'goddaughter', 'Female', 'sister-', 'girls--', 'herself', 'stepmothers', 'woman--', 'She', 'Stepdaughter', 'grandmotherly', 'girls-', 'females', 'Madwoman', 'granddaughter', 'sister--', 'birthmother', 'motherinlaw', 'Tgirl', 'godmother', 'Aunt', 'Grandmothers', 'Shemale', 'everywoman', 'girl-', 'woman-', 'grandmothers', 'aunts', 'daughtersinlaw', 'yourwife', "girls'll", 'sisters', 'daughter-', 'Amale', 'mother`s', 'madwoman', 'womans', 'mother-', 'Awoman', 'daughterinlaw', '/female', 'babygirl', 'egirl', 'girl--', 'stepmother', 'grandaughter', "mother'll", 'woman`s', 'sisterinlaw', 'grandmom', 'grandma', 'Granddaughter', 'auntie', 'womanly', 'Herself', 'niece', 'cavewoman', 'stepdaughter', "mother'd", 'godmothers', 'daughter--', 'Grandmother', 'Smother']
    assert lower
    return Query([lower_wordlist(targets_1), lower_wordlist(targets_2)],
                 [lower_wordlist(attributes_1), lower_wordlist(attributes_2)],
                 ['exp. career, math, and science', 'exp. family and arts'],
                 ['exp. male', 'exp. female'])


def winobias(lower=True):
    # MIT License
    #
    # Copyright (c) 2020 Natural Language Processing @UCLA
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    targets_1 = [
      "driver",
      "supervisor",
      "janitor",
      "cook",
      "mover",
      "laborer",
      "construction worker",
      "chief",
      "developer",
      "carpenter",
      "manager",
      "lawyer",
      "farmer",
      "salesperson",
      "physician",
      "guard",
      "analyst",
      "mechanic",
      "sheriff",
      "CEO",
    ]
    if lower:
        targets_1 = lower_wordlist(targets_1)
    targets_2 = [
      "attendant",
      "cashier",
      "teacher",
      "nurse",
      "assistant",
      "secretary",
      "auditor",
      "cleaner",
      "receptionist",
      "clerk",
      "counselor",
      "designer",
      "hairdresser",
      "writer",
      "housekeeper",
      "baker",
      "accountant",
      "editor",
      "librarian",
      "tailor",
    ]
    attributes_1 = ["he", "him"]
    attributes_2 = ["she", "her"]
    return Query(target_sets=[targets_1, targets_2],
                 attribute_sets=[attributes_1, attributes_2],
                 target_sets_names=['stereotypical male jobs', 'stereotypical female jobs'],
                 attribute_sets_names=['male', 'female'])


def winobias_exp(lower=True):
    targets_1 = ['driver', 'supervisor', 'janitor', 'cook', 'mover', 'laborer', 'construction worker', 'chief', 'developer', 'carpenter', 'manager', 'lawyer', 'farmer', 'salesperson', 'physician', 'guard', 'analyst', 'mechanic', 'sheriff', 'CEO', 'managers', 'stonemason', 'develo', 'Sheriffâ\x80\x99s', 'salespersons', 'labourers', 'farmers', 'VP', 'Atorney', 'guardsmen', 'cooks', 'sheriffdom', 'Sheriffs', "driver's", 'analysts', 'physiatrist', 'labor', 'strategist', 'anaesthesiologist', 'Busdriver', 'lawyer--', 'technician', 'sheriffs', 'Janitor', 'carpenters', 'managerial', 'CTO', 'blacksmith', 'warchief', 'constructional', 'Developer', 'supervise', 'lawyered', 'Salesperson', 'CFO', 'blacksmiths', 'Mover', 'mechanicals', 'ophthalmologist', 'analyser', 'guards', 'sawyer', 'farmhands', 'movers', 'foreman', 'driverâ\x80\x99s', 'CMO', 'constructing', 'lawyers', 'janitorial', 'shoemaker', 'Carpenter', 'anesthesiologist', 'develope', 'salespeople', 'supervises', 'chiefest', 'Chairman', 'attorney', 'salesy', 'cabdriver', 'carpentry', 'constructionist', 'construction', '-Sheriff', 'Domnitor', 'craftsman', 'clinician', 'solicitor', 'Sheriff', 'labored', 'cryptanalyst', 'mechano', 'telemarketer', 'mechanician', 'lawyerly', 'supervisory', 'Lawyer', 'Strategist', 'paediatrician', 'metaphysician', 'mechanics', 'Undersheriff', 'supervisorial', 'sherif', 'cardiologist', 'Chief', 'JDeveloper', 'supervisors', 'Redguard', 'janitors', 'managership', 'laboured', 'mechanically', 'COO', 'GoMechanic', 'Supervisor', 'farmhand', 'farming', 'rancher', 'develop-', 'Rearguard', 'sheriffdoms', 'earthmover', 'developers', 'attorneys', 'micromanager', 'farm', 'analyste', 'farmworker', 'labors', 'mechanical', 'laborers', 'proctologist', 'Warchief', 'drivers', 'cookii', 'construction-', 'labourer', 'labore', 'labours', 'tradesperson', 'C.E.O.s', 'C.E.O.', 'farmed', '-Chief', 'supervising', 'guardsman', 'chieh', 'Analyst', 'woodworker', 'analystes' 'moveth', 'move-', 'move--', 'development-', 'development', 'developement', 'analysiert', 'moving', 'carpenteri', 'driv']
    targets_2 = ['attendant', 'cashier', 'teacher', 'nurse', 'assistant', 'secretary', 'auditor', 'cleaner', 'receptionist', 'clerk', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'accountant', 'editor', 'librarian', 'tailor', 'accountants', 'assistants', 'libraries', 'Counselors', 'counseled', 'designers', 'salon', 'attendent', 'internist', 'archivist', 'Auditor', 'undersecretary', 'Undersecretary', 'Hairdressers', 'teacherâ\x80\x99s', 'clerked', 'Accountant', 'writerly', 'stylist', 'librarianship', 'nursed', 'retailor', 'nursey', 'library', 'Receptionist', 'clerks', 'counsellors', 'cleaners', 'secretarial', 'attendants', 'Attendants', 'coeditor', 'secretaries', 'coordinator', 'hairdressing', 'teacheth', 'Housekeeper', 'assister', 'accounting', 'editor(s', 'Secretary', 'interventionist', 'hairdressers', 'bakery', 'teacher-', 'counselled', '-Teacher', 'Cashier', 'editorial', 'copyeditor', 'Librarian', 'Attendant', 'bakeri', 'editors', 'teacher(s', 'bakers', 'bakeshop', 'cowriter', 'schoolteacher', 'nurses', 'librarians', 'Accountants', 'secretariats', 'Housekeepers', 'Clerk', 'auditors', 'assistantships', 'midwife', 'counselors', 'hairstylist', 'housekeeping', 'teacher--', 'tailoring', 'secretariat', 'counsellor', 'Counselor', 'tailormade', 'secretaryship', 'Secretaryship', 'audit', 'Auditors', 'audits', 'audited', 'pediatrician', 'midwifes', 'Assistant', 'assistantship', 'Designer', 'tailored', 'tailors', 'housecleaner', 'Librarianship', 'editorialist', 'Editor', 'subeditor', 'Hairdresser', 'Cashiers', 'secretario', 'counseling', 'cashiers', 'bakey', 'teachers&rsquo', 'accountancy', 'Receptionists', 'receptionists', 'assistances', 'assisté', 'housekeepers', 'nursie', 'cleans', 'teachin', 'clean', 'cleanin', 'cleaning', 'design.|', 'design', 'design-', 'bespoke', 'counsels', 'clerkship']
    attributes_1 = ['he', 'him', 'himselfe', 'He', 'himself-', 'himself--', 'himself']
    attributes_2 = ['she', 'her', 'yourwife', 'motherinlaw', 'hers', 'mother', 'everywoman', 'She', 'Herself', 'woman', 'stepmother', 'herself']
    assert lower
    return Query([lower_wordlist(targets_1), lower_wordlist(targets_2)],
                 [lower_wordlist(attributes_1), lower_wordlist(attributes_2)],
                 ['exp. stereotypical male jobs', 'exp. stereotypical female jobs'],
                 ['exp. male', 'exp. female'])


def winobias_rev(lower=True):
    winobias_query = winobias(lower)
    return Query(target_sets=winobias_query.attribute_sets,
                 attribute_sets=winobias_query.target_sets,
                 target_sets_names=winobias_query.attribute_sets_names,
                 attribute_sets_names=winobias_query.target_sets_names)


def hatespeech_gender(lower=True):
    return Query(target_sets=[["he's", 'boy', 'he', 'him', 'king', "men's",
                               'gentleman', 'husband'],
                              ["she's", 'girl', 'she', 'her', 'queen',
                               "women's", 'lady', 'wife']],
                 attribute_sets=[['evil', 'fuckin', 'idiot', 'niggas', 'fuck',
                                  'stupid', 'hate', 'nigga'],
                                 ['thanks', 'win', 'long', '–', 'liked', 'top',
                                  'star', 'national']],
                 target_sets_names=['male', 'female'],
                 attribute_sets_names=['hateful', 'unhateful'])


def hatespeech_gender_exp(lower=True):
    targets_1 = ["he's", 'boy', 'he', 'him', 'king', "men's", 'gentleman', 'husband', "he's-", 'himself-', 'kingship', 'gentlemanly', 'man', 'kings', 'princes', 'prince', 'husbandly', 'gentlemans', 'Gentlemans', 'Gentlemanly', 'Gentleman', 'gentlemen', 'gentlemen--', 'gent', 'princelings', 'princeling', 'princedom', 'himselfe', 'himself', 'himself--', 'Dibnah', 'husband-', 'husband--', "husband'll", 'He', "Men's", "He's--", "he's--", 'guy']
    targets_2 = ["she's", 'girl', 'she', 'her', 'queen', "women's", 'lady', 'wife', "She's-", 'women-', 'womens', 'daughter--', 'queef', 'wife--', 'women', 'women’s', 'woman', 'herself', "girls'll", 'motherinlaw', 'women&39;s', 'lady--', 'ladys', 'babygirl', "she's--", 'lady-', "lady's", 'She', 'girls--', 'women--', 'she--', 'wome', 'girl--', 'yourwife', "She's--", 'princess', "She's", 'egirl', 'wifey', 'Tgirl', 'everywoman', 'queenie', 'daughter', 'girl-', "M'lady", 'stepmother', 'wife-', 'women`s', "she'd", 'hers', 'mother', 'mywife', 'girls-', 'queens', 'daughter-', "she's-", 'Herself', 'womenfolk', '-women', 'queenly']
    attributes_1 = ['evil', 'fuckin', 'idiot', 'niggas', 'fuck', 'stupid', 'hate', 'nigga', 'stupider', 'disgusts', 'fool', 'fucking--', 'fucking-', "fuckin'-", "fuckin'--", 'fucka', 'evildoers', 'evildoer', 'devil', 'demon', 'demons', 'dumb', "fuck'd", 'Fuckin', 'hater', 'hate--', 'fuck-', '-Asshole', 'dislike', 'Niggas', 'niggah', 'fuckwad', 'fuckyou', 'goddamn', 'nigger', 'niggers', 'niggaz', 'malevolent', 'hates', 'Motherfuckin', 'motherfuckin', 'imbecile', '-Fuckin', 'ldiot', 'idiots', 'asshole', 'fucko', 'hateth', 'loathe', 'fuckwit', 'fuck--', 'ridiculous', '-Stupid', 'faggoty', 'faggot', 'stupid--', 'stupid-', 'evils', 'moron', 'idiota', 'faggots', 'fuckbag', 'idiotic', 'crazy', 'stupido', 'hateful', '-Motherfucker', 'despise', 'silly', 'vengeful', 'fucktoy']
    attributes_2 = ['thanks', 'win', 'long', '–', 'liked', 'top', 'star', 'national', 'star2', 'star1', 'star3', '5star', 'stars5', 'starsFive', 'longneck', 'likethe', '-thanks', 'Thanks', 'nationalising', 'long--', 'victoza', 'Vocale', 'top-3', 'ndash', 'top-100', 'wins', 'likee', 'stars', 'nationalen', '.Thanks', 'ndash;-', 'thankyou', 'Vocab', 'bottommost', 'FISE', 'lnternational', 'won', 'Subnational', 'subnational', '--Thanks', 'Supranational', 'starlet', 'nationales', 'winn', 'winning', 'thankyouso', 'tops', 'Top', 'top-', '(', ')', 'bottomline', 'nationalisation', 'winnin', 'longe', 'Trupp', 'star-', 'longwing', 'Remix', 'victory', 'longish', 'bottom', 'longarm', 'longclaw', 'longum', 'longfin', 'long-', 'thanks|', 'thank--', 'thanks!|', 'Lodash', 'loved', 'likea', 'Liked', 'defeat', '-National', 'clinch', 'international', 'Impr', 'highest', 'topmost', 'star--', 'victor']
    assert lower
    return Query([lower_wordlist(targets_1), lower_wordlist(targets_2)],
                 [lower_wordlist(attributes_1), lower_wordlist(attributes_2)],
                 ['exp. male', 'exp. female'],
                 ['exp. hateful', 'exp. unhateful'])


def hatespeech_race(lower=True):
    return Query([['amazing', 'automatically', 'anyone', 'awesome', 'nice', 'seeing', 'power', "here's", 'easter', "aren't", 'series', 'photos', 'less', '👍', 'weeks'],
                  ['goin', 'ion', 'stans', 'females', 'fa', 'bruh', 'bout', 'nerves', "ain't", 'yall', 'aint', 'lil', 'mama', 'sis', 'tryna']],
                 [['ready', 'liked', 'latest', 'social', '–', 'pretty', 'excited', 'light', 'favorite', 'far', 'perfect', 'easy', 'public', 'gemini', 'following', 'success', 'playlist', 'blue', 'virgo'],
                  ['slut', 'asshole', 'moron', 'fucks', 'bitches', 'bitch', 'fuckin', 'stupid', 'idiot', 'fucking', 'stressing', 'fucked', 'fuck', 'ugly', 'idiots', '😒', 'bullshit', 'bastard', 'shorty']],
                 ['white', 'african-american'], ['unhateful', 'hateful'])
