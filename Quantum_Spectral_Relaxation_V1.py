# Quantum_Spectral_Relaxation



from pathlib import Path
from dataclasses import dataclass
import csv
import itertools
import math
import random

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import logsumexp
from qiskit.quantum_info import DensityMatrix, Kraus

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4724_k93.csv"),
    Path("/data/loto7_4724_k93_loto_2984.csv"),
    Path("/data/loto7_4724_k93_loto_plus_1740.csv"),
)



N = 39
K = 7

# Brzine su deterministicki kandidati.
# Njihovi doprinosi u zajednickoj raspodeli uce se iz podataka.
MEMORY_RATES = np.array([0.015, 0.05, 0.15, 0.40])

# Parametar propagacije bira se hronoloskom validacijom.
PROPAGATION_TIMES = (0.75, 2.0)
DEPHASING = 0.15

OPTIMALITY_TOLERANCE = 1e-6
MAX_CONSECUTIVE = 4
DIAGNOSTICS = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_draws(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        except csv.Error:
            dialect = csv.excel

        rows = []
        for line_number, row in enumerate(csv.reader(handle, dialect), 1):
            fields = [value.strip() for value in row if value.strip()]
            if not fields:
                continue

            try:
                values = [int(value) for value in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: red {line_number} nije skup celih brojeva."
                ) from exc

            require(
                len(values) == K
                and len(set(values)) == K
                and min(values) >= 1
                and max(values) <= N,
                f"{path.name}: neispravna sedmorka u redu {line_number}.",
            )
            rows.append(sorted(value - 1 for value in values))

    require(len(rows) >= 100, f"{path.name}: potrebno je najmanje 100 kola.")
    return np.asarray(rows, dtype=np.int64)


def draw_amplitudes(draws, n, k):
    amplitudes = np.zeros((len(draws), n), dtype=np.float64)
    amplitudes[np.arange(len(draws))[:, None], draws] = 1.0 / math.sqrt(k)
    return amplitudes


@dataclass
class SpectralSystem:
    energies: np.ndarray
    vectors: np.ndarray
    reference: np.ndarray
    n: int
    k: int


def learn_spectral_system(training_draws, n=N, k=K):
    """
    Hamiltonijan se procenjuje samo iz trening dela.

    Realni simetricni deo opisuje uzajamnu strukturu prelaza.
    Imaginarni antisimetricni deo cuva njihov vremenski smer.
    Brojcana udaljenost oznaka 1..39 se ne koristi.
    """
    require(len(training_draws) >= 3, "Premalo trening kola.")
    amplitudes = draw_amplitudes(training_draws, n, k)

    left = amplitudes[:-1]
    right = amplitudes[1:]
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    transitions = left.T @ right / len(left)

    hamiltonian = (
        0.5 * (transitions + transitions.T)
        + 0.5j * (transitions - transitions.T)
    )
    hamiltonian -= np.eye(n) * np.trace(hamiltonian) / n

    require(
        np.max(np.abs(hamiltonian - hamiltonian.conj().T)) < 1e-12,
        "Hamiltonijan nije hermitski.",
    )

    energies, vectors = np.linalg.eigh(hamiltonian)
    radius = float(np.max(np.abs(energies)))

    if radius > 1e-14:
        energies = energies / radius

    require(
        np.max(np.abs(vectors.conj().T @ vectors - np.eye(n))) < 1e-10,
        "Sopstveni vektori nisu ortonormirani.",
    )

    # Pocetno mesovito stanje iz trening amplituda.
    # Ne postavlja se normalna ili uniformna raspodela ishoda.
    reference_native = amplitudes.T @ amplitudes / len(amplitudes)
    reference = vectors.conj().T @ reference_native @ vectors
    reference = 0.5 * (reference + reference.conj().T)

    require(
        abs(np.trace(reference).real - 1.0) < 1e-10,
        "Pocetno kvantno stanje nije normalizovano.",
    )
    require(
        np.min(np.linalg.eigvalsh(reference)) >= -1e-10,
        "Pocetno kvantno stanje nije pozitivno.",
    )

    return SpectralSystem(energies, vectors, reference, n, k)


def channel_factor(system, propagation_time):
    """
    Tacno resenje u energetskoj bazi:

    E(rho)_ab =
        exp[-i*t*(Ea-Eb) - gamma*t*(Ea-Eb)^2] * rho_ab

    Generator:
        -i[H,rho] - gamma[H,[H,rho]]

    Kanal je potpuno pozitivan i cuva trag.
    """
    gap = system.energies[:, None] - system.energies[None, :]
    return np.exp(
        -1j * propagation_time * gap
        - DEPHASING * propagation_time * gap * gap
    )


def simulate_history(draws, system, propagation_time):
    """
    Za svaki posmatrani red t:

      rho_t = (1-eta) E(rho_{t-1}) + eta |psi_t><psi_t|
      predikcija narednog kola = E(rho_t)

    Parametri sistema ostaju fiksni tokom simulacije.
    Red t+1 ne ucestvuje u stanju za predikciju tog reda.
    """
    n, k = system.n, system.k
    rates = MEMORY_RATES
    components = len(rates)
    factor = channel_factor(system, propagation_time)
    vectors = system.vectors

    states = np.repeat(system.reference[None, :, :], components, axis=0)
    all_probabilities = np.empty((len(draws), components, n))

    # Ovo je promena iz energetske u izvornu bazu merenja.
    # Nije naucena izlazna projekcija.
    measurement = (
        vectors[:, :, None] * vectors.conj()[:, None, :]
    ).reshape(n, n * n)

    for t, draw in enumerate(draws):
        amplitude = vectors[draw, :].conj().sum(axis=0) / math.sqrt(k)
        observation = np.outer(amplitude, amplitude.conj())

        states = (
            (1.0 - rates[:, None, None]) * states * factor[None, :, :]
            + rates[:, None, None] * observation[None, :, :]
        )
        prediction = states * factor[None, :, :]

        diagonal = (
            prediction.reshape(components, n * n) @ measurement.T
        )
        require(
            np.max(np.abs(diagonal.imag)) < 1e-9,
            "Dijagonala matrice gustine nije realna.",
        )
        probabilities = diagonal.real

        require(
            probabilities.min() >= -1e-10,
            "Kvantni kanal je dao negativnu verovatnocu.",
        )
        # Iskljucivo uklanjanje negativne greske zaokruzivanja.
        probabilities = np.maximum(probabilities, 0.0)
        traces = probabilities.sum(axis=1)
        require(
            np.max(np.abs(traces - 1.0)) < 1e-8,
            "Kvantni kanal ne cuva trag.",
        )
        probabilities /= traces[:, None]
        all_probabilities[t] = probabilities

    return all_probabilities


def elementary_symmetric(probabilities, k):
    """Tacna normalizacija za skupove od k razlicitih elemenata."""
    shape = probabilities.shape[:-1]
    coefficients = np.zeros(shape + (k + 1,))
    coefficients[..., 0] = 1.0

    for number in range(probabilities.shape[-1]):
        for degree in range(min(k, number + 1), 0, -1):
            coefficients[..., degree] += (
                probabilities[..., number] * coefficients[..., degree - 1]
            )

    return coefficients[..., k]


def component_log_probabilities(probabilities, targets):
    """
    Sedam registara imaju isto komponentno stanje.
    Uslovljavanje na razlicite rezultate i uklanjanje redosleda daje:

      P_r(S) = product(p_ri, i in S) / e_k(p_r)

    Faktor k! se skracuje. Nema sortirane autoregresije.
    """
    k = targets.shape[1]
    normalizer = elementary_symmetric(probabilities, k)
    require(
        np.all(np.isfinite(normalizer)) and np.all(normalizer > 0.0),
        "Nulta ili neispravna normalizacija cele kombinacije.",
    )

    selected = np.take_along_axis(
        probabilities, targets[:, None, :], axis=2
    )
    with np.errstate(divide="ignore"):
        log_selected = np.log(selected)
        result = log_selected.sum(axis=2) - np.log(normalizer)

    require(
        np.all(np.any(np.isfinite(result), axis=1)),
        "Posmatrana kombinacija nema podrsku ni u jednoj komponenti.",
    )
    return result


def mixture_objective(weights, scaled):
    mass = scaled @ weights
    if np.any(mass <= 0.0):
        return np.inf, np.zeros_like(weights)

    value = -float(np.mean(np.log(mass)))
    gradient = -np.mean(scaled / mass[:, None], axis=0)
    return value, gradient


def optimality_gap(weights, gradient):
    """
    Frank-Wolfe jaz za konveksni problem na simpleksu.
    Daje gornju granicu preostale razlike funkcije cilja do optimuma.
    """
    return max(
        0.0,
        float(np.dot(weights, gradient) - np.min(gradient)),
    )


def fit_mixture(log_probabilities):
    row_maximum = np.max(log_probabilities, axis=1)
    scaled = np.exp(log_probabilities - row_maximum[:, None])
    components = scaled.shape[1]

    # Samo pocetna tacka optimizacije, nije zadati konacni model.
    initial = np.full(components, 1.0 / components)

    constraint = {
        "type": "eq",
        "fun": lambda w: float(w.sum() - 1.0),
        "jac": lambda w: np.ones_like(w),
    }

    for attempt in range(2):
        result = minimize(
            mixture_objective,
            initial,
            args=(scaled,),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * components,
            constraints=constraint,
            options={
                "ftol": 1e-13,
                "maxiter": 1500,
                "disp": False,
            },
        )

        weights = np.asarray(result.x, dtype=float)
        require(np.all(np.isfinite(weights)), "Nekonacne tezine memorije.")
        require(
            weights.min() >= -1e-9 and abs(weights.sum() - 1.0) < 1e-7,
            "Optimizacija nije sacuvala simpleks.",
        )
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum()

        value, gradient = mixture_objective(weights, scaled)
        gap = optimality_gap(weights, gradient)

        if np.isfinite(value) and gap <= OPTIMALITY_TOLERANCE:
            return weights, gap

        initial = weights

    raise RuntimeError(
        "Ucenje memorije nije proslo proveru optimalnosti; "
        f"jaz={gap:.6g}; razlog={result.message}"
    )


def mixture_nll(log_probabilities, weights):
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0.0
    log_weights[positive] = np.log(weights[positive])
    values = logsumexp(log_probabilities + log_weights[None, :], axis=1)
    require(np.all(np.isfinite(values)), "Nekonacan NLL mesavine.")
    return -float(values.mean())


def combination_probability(probabilities, weights, combination):
    normalizer = elementary_symmetric(probabilities, len(combination))
    products = np.prod(probabilities[:, combination], axis=1)
    return float(np.sum(weights * products / normalizer))


@njit(cache=False)
def exhaustive_maximum(probabilities, coefficients, k, maximum_run):
    """
    Leksikografski prolaz kroz sve podskupove velicine k.
    Parcijalni proizvodi se osvezavaju samo od promenjene pozicije.
    Nema slucajnog uzorkovanja ni beam pretrage.
    """
    components, n = probabilities.shape
    combination = np.arange(k, dtype=np.int64)
    products = np.ones((k + 1, components))

    for depth in range(k):
        for component in range(components):
            products[depth + 1, component] = (
                products[depth, component]
                * probabilities[component, combination[depth]]
            )

    best_probability = -1.0
    best_combination = np.zeros(k, dtype=np.int64)
    visited = 0

    while True:
        visited += 1
        allowed = True
        run = 1

        for position in range(1, k):
            if combination[position] == combination[position - 1] + 1:
                run += 1
                if run > maximum_run:
                    allowed = False
                    break
            else:
                run = 1

        if allowed:
            probability = 0.0
            for component in range(components):
                probability += (
                    coefficients[component] * products[k, component]
                )

            if probability > best_probability:
                best_probability = probability
                best_combination[:] = combination

        pivot = k - 1
        while pivot >= 0 and combination[pivot] == n - k + pivot:
            pivot -= 1

        if pivot < 0:
            break

        combination[pivot] += 1
        for position in range(pivot + 1, k):
            combination[position] = combination[position - 1] + 1

        for depth in range(pivot, k):
            for component in range(components):
                products[depth + 1, component] = (
                    products[depth, component]
                    * probabilities[component, combination[depth]]
                )

    return best_combination, best_probability, visited


def decode(probabilities, weights, k=K, maximum_run=MAX_CONSECUTIVE):
    # Skaliranje pojedinacne komponente se skracuje sa normalizacijom.
    # Time se smanjuje opasnost od potkoracenja pri proizvodima.
    scales = probabilities.max(axis=1)
    require(np.all(scales > 0.0), "Prazna komponenta raspodele.")
    scaled = probabilities / scales[:, None]
    normalizer = elementary_symmetric(scaled, k)
    require(np.all(normalizer > 0.0), "Komponenta nema dovoljno razlicitih brojeva.")

    coefficients = weights / normalizer
    require(
        np.all(np.isfinite(coefficients)),
        "Numericki neispravna normalizacija dekodera.",
    )

    combination, probability, visited = exhaustive_maximum(
        scaled, coefficients, k, maximum_run
    )

    require(
        visited == math.comb(probabilities.shape[1], k),
        "Dekoder nije obisao sve kombinacije.",
    )
    require(
        probability > 0.0
        and len(set(combination.tolist())) == k
        and np.all(np.diff(combination) > 0),
        "Neispravan rezultat dekodiranja.",
    )

    checked = combination_probability(probabilities, weights, combination)
    require(
        abs(checked - probability) <= 1e-10 * max(checked, probability, 1e-300),
        "Ocena maksimuma nije usaglasena sa raspodelom.",
    )
    return combination, probability


def qiskit_channel_check(system, propagation_time):
    """Nezavisna provera analitickog kanala preko Kraus operatora."""
    n = system.n
    factor = channel_factor(system, propagation_time)
    eigenvalues, eigenvectors = np.linalg.eigh(factor)

    require(
        eigenvalues.min() >= -1e-10,
        "Jezgro dekoherencije nije pozitivno.",
    )

    operators = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= 0.0:
            continue
        diagonal = math.sqrt(eigenvalue) * eigenvectors[:, index]
        native = (
            system.vectors * diagonal[None, :]
        ) @ system.vectors.conj().T

        operator = np.zeros((64, 64), dtype=np.complex128)
        operator[1:n + 1, 1:n + 1] = native
        operators.append(operator)

    unused = np.eye(64, dtype=np.complex128)
    unused[1:n + 1, 1:n + 1] = 0.0
    operators.append(unused)

    completeness = sum(operator.conj().T @ operator for operator in operators)
    require(
        np.max(np.abs(completeness - np.eye(64))) < 1e-9,
        "Kraus operatori ne cuvaju trag.",
    )

    reference_native = (
        system.vectors @ system.reference @ system.vectors.conj().T
    )
    embedded = np.zeros((64, 64), dtype=np.complex128)
    embedded[1:n + 1, 1:n + 1] = reference_native

    actual = np.asarray(
        DensityMatrix(embedded).evolve(Kraus(operators)).data
    )
    expected = (
        system.vectors
        @ (factor * system.reference)
        @ system.vectors.conj().T
    )

    require(
        np.max(np.abs(actual[1:n + 1, 1:n + 1] - expected)) < 1e-9,
        "Qiskit i analiticki kanal se ne poklapaju.",
    )
    require(
        np.min(np.linalg.eigvalsh(actual)) >= -1e-9
        and abs(np.trace(actual).real - 1.0) < 1e-9,
        "Qiskit izlaz nije ispravno kvantno stanje.",
    )


def longest_run(combination):
    run = best = 1
    for a, b in zip(combination[:-1], combination[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def internal_checks():
    n, k = 7, 3
    combinations = list(itertools.combinations(range(n), k))
    draws = np.asarray(
        [combinations[(11 * t + t // 4) % len(combinations)] for t in range(45)],
        dtype=np.int64,
    )

    system = learn_spectral_system(draws[:28], n, k)
    qiskit_channel_check(system, 0.75)
    probabilities = simulate_history(draws, system, 0.75)

    require(
        np.max(np.abs(probabilities.sum(axis=2) - 1.0)) < 1e-10,
        "Komponente nisu normalizovane.",
    )

    logs = component_log_probabilities(probabilities[:-1], draws[1:])
    weights, gap = fit_mixture(logs[:27])
    require(gap <= OPTIMALITY_TOLERANCE, "Provera optimalnosti nije prosla.")

    next_probabilities = probabilities[-1]
    total = sum(
        combination_probability(next_probabilities, weights, item)
        for item in combinations
    )
    require(abs(total - 1.0) < 1e-10, "Cela raspodela nije normalizovana.")

    decoded, score = decode(
        next_probabilities, weights, k=k, maximum_run=2
    )
    expected = max(
        combination_probability(next_probabilities, weights, item)
        for item in combinations
        if longest_run(item) <= 2
    )
    require(
        abs(score - expected) < 1e-11,
        "Dekoder nije pronasao pravi maksimum.",
    )
    require(longest_run(decoded) <= 2, "Ogranicenje uzastopnih nije ispravno.")

    reversed_order = tuple(reversed(combinations[5]))
    require(
        abs(
            combination_probability(next_probabilities, weights, combinations[5])
            - combination_probability(next_probabilities, weights, reversed_order)
        ) < 1e-13,
        "Verovatnoca zavisi od redosleda zapisa.",
    )

    modified = draws.copy()
    modified[35:] = draws[35:][::-1]
    second = simulate_history(modified, system, 0.75)
    require(
        np.array_equal(probabilities[:35], second[:35]),
        "Buduci podaci menjaju ranija stanja memorije.",
    )

    # Provera analitickog gradijenta mesavine u unutrasnjosti simpleksa.
    row_maximum = logs.max(axis=1)
    scaled = np.exp(logs - row_maximum[:, None])
    point = np.full(len(MEMORY_RATES), 1.0 / len(MEMORY_RATES))
    _, gradient = mixture_objective(point, scaled)
    step = 1e-6

    for index in range(len(point)):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        vp, _ = mixture_objective(plus, scaled)
        vm, _ = mixture_objective(minus, scaled)
        require(
            abs((vp - vm) / (2 * step) - gradient[index]) < 1e-6,
            "Gradijent mesavine nije prosao proveru.",
        )

    # Kanal relaksacije mora kontrahovati razliku pocetnih stanja.
    factor = channel_factor(system, 0.75)
    difference = system.reference - np.diag(np.diag(system.reference))
    initial_norm = np.linalg.norm(difference)

    for rate in MEMORY_RATES:
        evolved = difference.copy()
        for _ in range(8):
            evolved = (1.0 - rate) * factor * evolved
        require(
            np.linalg.norm(evolved)
            <= (1.0 - rate) ** 8 * initial_norm + 1e-12,
            "Relaksacija ne kontrahuje pocetnu razliku.",
        )


def fit_stage(draws, training_end, propagation_time):
    """
    training_end je broj trening prelaza.
    Trening ukljucuje redove 0..training_end.
    """
    system = learn_spectral_system(draws[:training_end + 1])
    probabilities = simulate_history(draws, system, propagation_time)
    logs = component_log_probabilities(probabilities[:-1], draws[1:])
    weights, gap = fit_mixture(logs[:training_end])
    return system, probabilities, logs, weights, gap


def process_csv(path):
    draws = read_draws(path)
    transitions = len(draws) - 1
    train_end = int(0.70 * transitions)
    validation_end = int(0.85 * transitions)

    # Jedan trening Hamiltonijan za sve validacione kandidate.
    training_system = learn_spectral_system(draws[:train_end + 1])
    best = None

    for propagation_time in PROPAGATION_TIMES:
        probabilities = simulate_history(
            draws[:validation_end + 1], training_system, propagation_time
        )
        logs = component_log_probabilities(
            probabilities[:-1], draws[1:validation_end + 1]
        )
        weights, gap = fit_mixture(logs[:train_end])
        validation = mixture_nll(logs[train_end:validation_end], weights)

        if best is None or validation < best[0]:
            best = (validation, propagation_time, weights)

    validation, propagation_time, validation_weights = best
    print(
        f"Validacija NLL: {validation:.6f} | "
        f"propagacija: {propagation_time:g}"
    )

    _, _, test_logs, test_weights, _ = fit_stage(
        draws, validation_end, propagation_time
    )
    test = mixture_nll(test_logs[validation_end:], test_weights)
    print(f"Test NLL: {test:.6f}")

    final_system, final_probabilities, _, final_weights, gap = fit_stage(
        draws, transitions, propagation_time
    )

    # Provera kanala i na konacnom Hamiltonijanu konkretnog CSV-a.
    qiskit_channel_check(final_system, propagation_time)

    next_probabilities = final_probabilities[-1]
    print(f"Pretraga NEXT: svih {math.comb(N, K)} kombinacija.")
    combination, probability = decode(next_probabilities, final_weights)
    next_values = tuple(int(number + 1) for number in combination)
    print(f"NEXT: {list(next_values)}")

    if DIAGNOSTICS:
        print(f"Doprinosi memorije: {np.round(final_weights, 6).tolist()}")
        print(f"Brzine memorije: {MEMORY_RATES.tolist()}")
        print(f"Jaz optimalnosti: {gap:.6g}")
        print(f"Verovatnoca kombinacije: {probability:.12g}")

    return path.name, next_values, next_probabilities, final_weights, draws


def compare_results(results):
    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            name_a, next_a, probabilities_a, weights_a, draws_a = results[left]
            name_b, next_b, probabilities_b, weights_b, draws_b = results[right]

            if next_a != next_b:
                continue

            same_data = (
                draws_a.shape == draws_b.shape
                and np.array_equal(draws_a, draws_b)
            )
            maximum_log_difference = 0.0

            for step in (1, 2, 4, 5, 7, 8, 10, 11):
                for start in range(N):
                    probe = tuple(
                        sorted((start + step * index) % N for index in range(K))
                    )
                    if len(set(probe)) != K:
                        continue

                    pa = combination_probability(probabilities_a, weights_a, probe)
                    pb = combination_probability(probabilities_b, weights_b, probe)

                    if pa > 0.0 and pb > 0.0:
                        maximum_log_difference = max(
                            maximum_log_difference,
                            abs(math.log(pa) - math.log(pb)),
                        )
                    elif pa != pb:
                        maximum_log_difference = math.inf

            print(
                f"PROVERA: isti NEXT za {name_a} i {name_b}; "
                f"isti podaci: {same_data}; "
                f"razlika log-verovatnoca na probama: "
                f"{maximum_log_difference:.6g}."
            )


def main():
    print("Quantum Spectral Relaxation V1.")
    print("Tacna dijagonalizacija; dekoherencija i naucena mesavina memorija.")
    print("Kvantno: amplitude, matrice gustine i lokalni CPTP kanali.")
    print("Klasicno: procena Hamiltonijana, ucenje mesavine i iscrpna pretraga.")
    print("Sedam 6-qubitnih registara; zajednicki izbor komponente memorije.")
    print("Tacan NLL cele sedmorke; bez nametnute termalne raspodele.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.")

    internal_checks()
    print("Interne provere: prosle.")

    results = []
    failures = 0

    for path in FILES:
        print()
        print(f"CSV: {path.name}")
        try:
            results.append(process_csv(path))
        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}")

    compare_results(results)

    if failures:
        print()
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")



"""
Quantum Spectral Relaxation V1.
Tacna dijagonalizacija; dekoherencija i naucena mesavina memorija.
Kvantno: amplitude, matrice gustine i lokalni CPTP kanali.
Klasicno: procena Hamiltonijana, ucenje mesavine i iscrpna pretraga.
Sedam 6-qubitnih registara; zajednicki izbor komponente memorije.
Tacan NLL cele sedmorke; bez nametnute termalne raspodele.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4724_k93.csv
Validacija NLL: 16.552501 | propagacija: 2
Test NLL: 16.553204
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [3, x, 9, y, 28, z, 37]

CSV: loto7_4724_k93_loto_2984.csv
Validacija NLL: 16.554912 | propagacija: 2
Test NLL: 16.554285
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [5, x, 11, y, 26, z, 35]

CSV: loto7_4724_k93_loto_plus_1740.csv
Validacija NLL: 16.553863 | propagacija: 2
Test NLL: 16.552632
Pretraga NEXT: svih 15380937 kombinacija.
NEXT: [1, x, 3, y, 17, z, 31]
"""



"""
Učenje koliko dugo različiti istorijski obrasci zadržavaju uticaj na NEXT. 

Oblast	                Konkretan doprinos
Tačna dijagonalizacija	Rastavlja konačni Hamiltonijan na sopstvena stanja i energije. Omogućava precizno računanje dinamike izabranog modela; sama ne određuje šta treba učiti.
Termalizacija	        Daje ideju relaksacije: neki tragovi početnog stanja brzo slabe, dok drugi traju duže. To je najzanimljiviji deo za predikciju.
Ergodičnost	            Povezuje dugoročne vremenske proseke sa prosekom raspodele, pod odgovarajućim uslovima. Ne znači nezavisnost kola niti garantuje brzo zaboravljanje istorije.

Veza spektra, dinamike i termalizacije.


Model sa naučenim komponentama memorije. 
Svako prethodno kolo menja unutrašnje stanje; različite komponente imaju različite naučene brzine slabljenja i moguće oscilacije. 
Iz njihovog zajedničkog stanja model predviđa raspodelu cele naredne sedmorke.

Model sam razlikuje kratkotrajne promene od dugotrajnijih obrazaca, umesto da svima nametnemo isti prozor ili unapred zadato opadanje.

Samo unitarno razvijanje konačnog zatvorenog sistema ne daje proizvoljno eksponencijalno zaboravljanje. 
Za takvu memoriju moramo jasno definisati otvoreni kvantni sistem ili praćenje njegovog podsistema.

Dijagonalizacija je računski alat, a ergodičnost pomoć za razumevanje ponašanja modela — ne tri odvojene V1.

V1 zasnovan na otvorenoj kvantnoj dinamici: 
model će učiti doprinos kratke i duge memorije, uz tačnu normalizaciju raspodele sedmorke. 
"""
