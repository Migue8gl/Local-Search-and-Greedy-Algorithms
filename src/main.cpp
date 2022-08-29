#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

vector<vector<double>> data_matrix;
vector<char> class_vector;
random_device r;
double Seed = 3812875771;

void readData(string file)
{
    vector<vector<string>> data_matrix_aux;
    string ifilename = file;
    ifstream ifile;
    istream *input = &ifile;

    ifile.open(ifilename.c_str());

    if (!ifile)
    {
        cerr << "Error opening file " << endl;
        exit(1);
    }

    string data;
    int cont = 0, cont_aux = 0;
    char aux;
    vector<string> aux_vector;
    bool finish = false;

    // Leo número de atributos y lo guardo en contador
    do
    {
        *input >> data;
        if (data == "@attribute")
            cont++;
    } while (data != "@data"); // A partir de aquí leemos datos

    data = "";

    // Mientras no lleguemos al final leemos datos
    while (!(*input).eof())
    {
        // Leemos caracter a caracter
        *input >> aux;

        /* Si hemos terminado una linea de datos la guardamos en la matrix de datos
        y reiniciamos el contador auxiliar (nos dice por qué dato vamos) */
        if (finish)
        {
            data_matrix_aux.push_back(aux_vector);
            aux_vector.clear();
            cont_aux = 0;
            finish = false;
        }

        /* Si hay una coma el dato ha terminado de leerse y lo almacenamos, en caso
        contrario seguimos leyendo caracteres y almacenandolos en data*/
        if (aux != ',' && cont_aux < cont)
        {
            data += aux;
            // Si hemos llegado al penultimo elemento hemos terminado
            if (cont_aux == cont - 1)
            {
                cont_aux++;
                aux_vector.push_back(data);
                data = "";
                finish = true;
            }
        }
        else
        {
            aux_vector.push_back(data);
            data = "";
            cont_aux++;
        }
    }

    vector<double> vect_aux;

    for (vector<vector<string>>::iterator it = data_matrix_aux.begin(); it != data_matrix_aux.end(); it++)
    {
        vect_aux.clear();
        for (vector<string>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            if (jt == it->end() - 1)
                class_vector.push_back((*jt)[0]);
            else
                vect_aux.push_back(stod(*jt));
        }
        data_matrix.push_back(vect_aux);
    }
}

void normalizeData(vector<vector<double>> &data)
{
    double item = 0.0;           // Característica individual
    double max_item = -999999.0; // Valor máximo del rango de valores
    double min_item = 999999.0;  // Valor minimo del rango de valores

    // Buscamos los máximos y mínimos
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
        {
            item = *jt;

            if (item > max_item)
                max_item = item;

            if (item < min_item)
                min_item = item;
        }

    // Normalizamos aplicando x_iN = (x_i - min) / (max - min)
    for (vector<vector<double>>::iterator it = data.begin(); it != data.end(); it++)
        for (vector<double>::iterator jt = it->begin(); jt != it->end(); jt++)
            *jt = (*jt - min_item) / (max_item - min_item);
}

pair<vector<vector<vector<double>>>, vector<vector<char>>> createPartitions()
{
    vector<vector<double>> data_m_aux = data_matrix;
    vector<char> class_v_aux = class_vector;

    // Mezclo aleatoriamente la matriz original
    srand(Seed);
    random_shuffle(begin(data_m_aux), end(data_m_aux));
    srand(Seed);
    random_shuffle(begin(class_vector), end(class_vector));

    const int MATRIX_SIZE = data_matrix.size();
    vector<vector<double>>::iterator it = data_m_aux.begin();
    vector<char>::iterator jt = class_v_aux.begin();

    // Particiones y puntero que las irá recorriendolas para insertar datos
    vector<vector<double>> g1, g2, g3, g4, g5, *g_aux;
    vector<char> g1c, g2c, g3c, g4c, g5c, *g_aux2;
    int cont = 0, cont_grupos = 0;
    bool salir = false;

    // Mientras no se hayan insertado todos los datos en todos los grupos
    while (cont != MATRIX_SIZE && cont_grupos < 5)
    {
        // Elegimos la partición que toque
        switch (cont_grupos)
        {
        case 0:
            g_aux = &g1;
            g_aux2 = &g1c;
            break;
        case 1:
            g_aux = &g2;
            g_aux2 = &g2c;
            break;
        case 2:
            g_aux = &g3;
            g_aux2 = &g3c;
            break;
        case 3:
            g_aux = &g4;
            g_aux2 = &g4c;
            break;
        case 4:
            g_aux = &g5;
            g_aux2 = &g5c;
            break;
        }

        // Vamos rellenando la partición pertinente
        for (int k = 0; k < MATRIX_SIZE / 5 && !salir; k++)
        {
            g_aux->push_back(*it);
            g_aux2->push_back(*jt);
            it++;
            jt++;
            cont++;

            /* Si estamos en el último grupo y quedan todavía elementos, seguir
            insertándolos en este último */
            if (cont_grupos == 4)
            {
                if (it != data_m_aux.end())
                    k--;
                else
                    salir = true;
            }
        }
        cont_grupos++;
    }
    vector<vector<vector<double>>> d = {g1, g2, g3, g4, g5};
    vector<vector<char>> c = {g1c, g2c, g3c, g4c, g5c};
    pair<vector<vector<vector<double>>>, vector<vector<char>>> partitions = make_pair(d, c);

    return partitions;
}

char KNN_Classifier(vector<vector<double>> &data, vector<vector<double>>::iterator &elem, vector<char> &elemClass, vector<double> &w)
{
    vector<double> distancia;
    vector<char> clases;
    vector<char>::iterator cl = elemClass.begin();
    vector<double>::iterator wi = w.begin();
    vector<double>::iterator ej;
    double sumatoria = 0;
    double dist_e = 0;
    int hola = 0, hola2 = 0;

    for (vector<vector<double>>::iterator e = data.begin(); e != data.end(); e++)
    {
        // Si el elemento es él mismo no calculamos distancia, pues es 0
        if (elem != e)
        {
            sumatoria = 0;
            ej = elem->begin();
            wi = w.begin();

            // Calculamos distancia de nuestro elemento con el resto
            for (vector<double>::iterator ei = e->begin(); ei != e->end() - 1; ei++)
            {
                sumatoria += *wi * pow(*ej - *ei, 2);
                ej++;
                wi++;
            }
            dist_e = sqrt(sumatoria);
            distancia.push_back(dist_e);
            clases.push_back(*cl);
        }
        cl++;
    }

    vector<double>::iterator it;
    vector<char>::iterator cl_dist_min = clases.begin();

    double distMin = 99999;
    char vecinoMasProxClass;

    // Nos quedamos con el que tenga minima distancia, es decir, su vecino más próximo
    for (it = distancia.begin(); it != distancia.end(); it++)
    {
        if (*it < distMin)
        {
            distMin = *it;
            vecinoMasProxClass = *cl_dist_min;
        }
        cl_dist_min++;
    }

    return vecinoMasProxClass;
}

double calculaAciertos(vector<vector<double>> &muestras, vector<char> &clases, vector<double> &w)
{
    double instBienClasificadas = 0.0;
    double numIntanciasTotal = muestras.size();
    char cl_1NN;
    vector<char>::iterator c_it = clases.begin();

    for (vector<vector<double>>::iterator it = muestras.begin(); it != muestras.end(); it++)
    {
        cl_1NN = KNN_Classifier(muestras, it, clases, w);

        if (cl_1NN == *c_it)
            instBienClasificadas += 1.0;
        c_it++;
    }

    return instBienClasificadas / numIntanciasTotal;
}

vector<double> buscaAmigo(vector<vector<double>> &muestra, vector<vector<double>>::iterator &item, vector<char> &itemClass)
{
    vector<pair<vector<double>, double>> amigos;
    vector<char> clases;
    vector<char>::iterator cl = itemClass.begin();
    char clase = '\0';
    double sumatoria = 0;
    double dist_e = 0;

    vector<double>::iterator ej;
    for (vector<vector<double>>::iterator e = muestra.begin(); e != muestra.end(); e++)
    {
        if (item != e)
        {
            sumatoria = 0;
            ej = item->begin();

            for (vector<double>::iterator ei = e->begin(); ei != e->end(); ei++)
            {
                sumatoria += pow(*ej - *ei, 2);
                ej++;
            }

            dist_e = sqrt(sumatoria);
            amigos.push_back(make_pair(*e, dist_e));
            clases.push_back(*cl);
            cl++;
        }
        else
        {
            clase = *cl;
            cl++;
        }
    }

    vector<pair<vector<double>, double>>::iterator it;
    vector<char>::iterator cl_amig = clases.begin();

    double distMin = 99999;
    vector<double> amigoMasProx;

    // Nos quedamos con el amigo más cercano
    for (it = amigos.begin(); it != amigos.end(); it++)
    {
        // Si la clase coincide es que son amigos
        if ((*it).second < distMin && clase == *cl_amig)
        {
            distMin = (*it).second;
            amigoMasProx = (*it).first;
        }
        cl_amig++;
    }

    return amigoMasProx;
}

vector<double> buscaEnemigo(vector<vector<double>> &muestra, vector<vector<double>>::iterator &item, vector<char> &itemClass)
{
    vector<pair<vector<double>, double>> enemigos;
    vector<char> clases;
    vector<char>::iterator cl = itemClass.begin();
    char clase = '\0';
    double sumatoria = 0;
    double dist_e = 0;

    vector<double>::iterator ej;
    for (vector<vector<double>>::iterator e = muestra.begin(); e != muestra.end(); e++)
    {
        if (item != e)
        {
            sumatoria = 0;
            ej = item->begin();

            for (vector<double>::iterator ei = e->begin(); ei != e->end(); ei++)
            {
                sumatoria += pow(*ej - *ei, 2);
                ej++;
            }
            dist_e = sqrt(sumatoria);
            enemigos.push_back(make_pair(*e, dist_e));
            clases.push_back(*cl);
            cl++;
        }
        else
        {
            clase = *cl;
            cl++;
        }
    }

    vector<pair<vector<double>, double>>::iterator it;
    vector<char>::iterator cl_enem = clases.begin();

    double distMin = 99999;
    vector<double> enemigoMasProx;

    // Nos quedamos con el enemigo más cercano
    for (it = enemigos.begin(); it != enemigos.end(); it++)
    {
        // Si la clase no coincide es que son enemigos
        if ((*it).second < distMin && clase != *cl_enem)
        {
            distMin = (*it).second;
            enemigoMasProx = (*it).first;
        }
        cl_enem++;
    }

    return enemigoMasProx;
}

vector<double> reliefAlg(vector<vector<double>> &muestra, vector<char> &muestra_clases)
{
    // Inicializamos el vector de pesos a 0
    vector<double> w(muestra.begin()->size(), 0.0);
    vector<double> amigo;
    vector<double> enemigo;
    vector<double>::iterator ei;
    vector<double>::iterator e_amg;
    vector<double>::iterator e_enem;
    double w_max = -99999;

    // Por cada elemento buscamos su amigo y enemigo más cercano
    for (vector<vector<double>>::iterator it = muestra.begin(); it != muestra.end(); it++)
    {
        // Buscamos amigo y enemigo del elemento
        amigo = buscaAmigo(muestra, it, muestra_clases);
        enemigo = buscaEnemigo(muestra, it, muestra_clases);

        // Nos posicionamos al principio del contenedor
        ei = it->begin();
        e_amg = amigo.begin();
        e_enem = enemigo.begin();

        // Por cada característica ajustamos unos pesos (w)
        for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
        {
            *wi += abs(*ei - *e_enem) - abs(*ei - *e_amg);
            if (*wi > w_max)
                w_max = *wi;
            ei++;
            e_enem++;
            e_amg++;
        }
    }

    // Normalizamos los pesos entre [0,1]
    for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
    {
        if (*wi < 0)
            *wi = 0.0;
        else
            *wi /= w_max;
    }

    return w;
}

vector<double> knnAlg(vector<vector<double>> &muestra, vector<char> &muestra_clases)
{
    // Inicializamos el vector de pesos a 1
    vector<double> w(muestra.begin()->size(), 1.0);
    return w;
}

vector<double> blAlg(vector<vector<double>> &muestra, vector<char> &muestra_clases)
{
    /* Creo un generador de números entre 0 y 1 con distribución uniforme
    de números reales */
    ;
    mt19937 eng(Seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    auto gen = [&dist, &eng]()
    {
        return dist(eng);
    };

    // Inizializo vector de pesos con el generador
    vector<double> w(muestra.begin()->size());
    generate(begin(w), end(w), gen);

    const int maxIter = 15000;
    const int maxVecin = 20;
    int cont = 0, vecinos = 0;
    double varianza = 0.3, alpha = 0.5;
    int numCaract = muestra.begin()->size();

    // Creo vector z y un generador de distribución normal
    vector<double> z(w.size());
    vector<double>::iterator z_it;
    normal_distribution<double> normal_dist(0.0, pow(varianza, 2));
    double s = r();
    mt19937 other_eng(s);
    auto genNormalDist = [&normal_dist, &other_eng]()
    {
        return normal_dist(other_eng);
    };

    double tasa_clas = 0;
    double tasa_red = 0;
    double fun_objetivo = 0;
    double cont_red = 0.0;
    double max_fun = -99999.0;
    double w_aux;

    // Mientras no se superen las iteraciones máximas o los vecinos permitidos
    while (vecinos < (maxVecin * numCaract) && cont < maxIter)
    {
        generate(begin(z), end(z), genNormalDist);
        z_it = z.begin();
        cont_red = 0;

        for (vector<double>::iterator it = w.begin(); it != w.end(); it++)
        {
            // Guardamos w original
            w_aux = *it;

            // Mutación normal
            *it += *z_it;

            if (*it < 0.1)
            {
                *it = 0;
                cont_red += 1.0;
            }
            else if (*it > 1)
            {
                *it = 1;
                cont_red += 1.0;
            }

            tasa_clas = calculaAciertos(muestra, muestra_clases, w);
            tasa_red = cont_red / w.size();
            fun_objetivo = alpha * tasa_clas + (1.0 - alpha) * tasa_red;

            // Si hemos mejorado el umbral a mejorar cambia, vamos maximizando la función
            if (fun_objetivo > max_fun)
            {
                max_fun = fun_objetivo;
                vecinos = 0;
            }
            else
            {
                // Si no hemos mejorado nos quedamos con la w anterior
                *it = w_aux;
                vecinos++;
            }
            z_it++;
        }
        cont++;
    }

    return w;
}

void execute(pair<vector<vector<vector<double>>>, vector<vector<char>>> &part, vector<double> (*alg)(vector<vector<double>> &, vector<char> &))
{
    vector<double> w;
    vector<vector<vector<double>>>::iterator data_test = part.first.begin();
    vector<vector<char>>::iterator class_test = part.second.begin();
    vector<vector<double>> aux_data_fold;
    vector<char> aux_class_fold;
    vector<vector<vector<double>>>::iterator it;
    vector<vector<char>>::iterator jt;

    double tasa_clas = 0;
    double tasa_red = 0;
    double agregado = 0;
    double alpha = 0.5;
    double cont_red = 0.0;
    double TS_media = 0, TR_media = 0, A_media = 0;
    int cont = 0;

    auto momentoInicio = high_resolution_clock::now();

    // Iteramos 5 veces ejecutando el algoritmo
    while (cont < 5)
    {
        jt = part.second.begin();
        aux_data_fold.clear();
        aux_class_fold.clear();
        cont_red = 0.0;

        // Creamos particiones train
        for (it = part.first.begin(); it != part.first.end(); it++)
        {
            // Si es una partición test no la añadimos a test
            if (it != data_test && jt != class_test)
            {
                aux_data_fold.insert(aux_data_fold.end(), (*it).begin(), (*it).end());
                aux_class_fold.insert(aux_class_fold.end(), (*jt).begin(), (*jt).end());
            }
            jt++;
        }

        // Ejecución del algoritmo
        auto partInicio = high_resolution_clock::now();
        w = alg(aux_data_fold, aux_class_fold);
        auto partFin = high_resolution_clock::now();

        for (vector<double>::iterator wi = w.begin(); wi != w.end(); wi++)
            if (*wi < 0.1)
            {
                cont_red += 1.0;
                *wi = 0.0;
            }

        tasa_clas = calculaAciertos(*data_test, *class_test, w);
        tasa_red = cont_red / w.size();
        agregado = alpha * tasa_clas + (1 - alpha) * tasa_red;

        milliseconds tiempo_part = duration_cast<std::chrono::milliseconds>(partFin - partInicio);

        cout << cont + 1 << "Tasa_clas: " << tasa_clas << endl;
        cout << cont + 1 << "Tasa_red: " << tasa_red << endl;
        cout << cont + 1 << "Agregacion: " << agregado << endl;
        cout << cont + 1 << "Tiempo_ejecucion: " << tiempo_part.count() << " miliseg\n\n";

        TS_media += tasa_clas;
        TR_media += tasa_red;
        A_media += agregado;

        cont++;
        data_test++;
        class_test++;
    }
    auto momentoFin = high_resolution_clock::now();

    milliseconds tiempo = duration_cast<std::chrono::milliseconds>(momentoFin - momentoInicio);

    cout << "Tasa_clas_media: " << TS_media / 5 << endl;
    cout << "Tasa_red_media: " << TR_media / 5 << endl;
    cout << "Agregacion_media: " << A_media / 5 << endl;
    cout << "Tiempo_ejecucion_medio: " << tiempo.count() << " miliseg";
}

int main(int nargs, char *args[])
{
    char *arg[4];
    string option;
    string path;

    for (int i = 1; i < nargs;)
    {
        option = args[i++];

        if (option == "1")
            path = "./Instancias_APC/spectf-heart.arff";

        else if (option == "2")
            path = "./Instancias_APC/parkinsons.arff";
        else if (option == "3")
            path = "./Instancias_APC/ionosphere.arff";
        else
        {
            cerr << "Error en parámetros" << endl;
            exit(1);
        }
    }

    readData(path);
    normalizeData(data_matrix);

    pair<vector<vector<vector<double>>>, vector<vector<char>>> part;
    part = createPartitions();

    cout << "Semilla: " << setprecision(10) << Seed << endl;

    cout << "\n-----ALGORITMO RELIEF-----\n";
    execute(part, reliefAlg);
    cout << "\n\n-----ALGORITMO BÚSQUEDA LOCAL-----\n";
    execute(part, blAlg);
    cout << "\n\n-----ALGORITMO 1-NN-----\n";
    execute(part, knnAlg);

    cout << endl
         << endl;
}
