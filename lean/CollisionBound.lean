import Std

/- A finite-sample counting certificate, not a claim of linguistic novelty.
   Each list is the recorded labels in one representation-equivalence class.
   A deterministic predictor chooses one label for that class. -/
namespace NormalizationAudit

def largest : List Nat → Nat
  | [] => 0
  | a :: xs => max a (largest xs)

theorem member_le_largest (xs : List Nat) (a : Nat) (h : a ∈ xs) :
    a ≤ largest xs := by
  induction xs with
  | nil => simp at h
  | cons x xs ih =>
    simp only [List.mem_cons] at h
    cases h with
    | inl h => subst a; exact Nat.le_max_left _ _
    | inr h => exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

theorem largest_attained (xs : List Nat) (h : xs ≠ []) :
    ∃ a ∈ xs, largest xs = a := by
  induction xs with
  | nil => contradiction
  | cons x xs ih =>
    cases xs with
    | nil => exact ⟨x, by simp, by simp [largest]⟩
    | cons y ys =>
      obtain ⟨a, ha, heq⟩ := ih (by simp)
      by_cases hx : a ≤ x
      · refine ⟨x, by simp, ?_⟩
        change max x (largest (y :: ys)) = x
        rw [heq, Nat.max_eq_left hx]
      · refine ⟨a, List.mem_cons.mpr (Or.inr ha), ?_⟩
        change max x (largest (y :: ys)) = a
        rw [heq, Nat.max_eq_right (by omega)]

def maxFrequency (xs : List Nat) : Nat :=
  largest (xs.map (fun a => xs.count a))

theorem count_le_maxFrequency (xs : List Nat) (prediction : Nat) :
    xs.count prediction ≤ maxFrequency xs := by
  by_cases h : prediction ∈ xs
  · exact member_le_largest _ _ (List.mem_map.mpr ⟨prediction, h, rfl⟩)
  · rw [List.count_eq_zero_of_not_mem h]
    exact Nat.zero_le _

theorem class_error_lower_bound (xs : List Nat) (prediction : Nat) :
    xs.length - maxFrequency xs ≤ xs.length - xs.count prediction := by
  have h := count_le_maxFrequency xs prediction
  omega

theorem class_error_bound_attainable (xs : List Nat) :
    ∃ prediction : Nat,
      xs.length - xs.count prediction = xs.length - maxFrequency xs := by
  by_cases h : xs = []
  · subst xs; exact ⟨0, rfl⟩
  · have hm : xs.map (fun a => xs.count a) ≠ [] := by simpa using h
    obtain ⟨n, hn, heq⟩ := largest_attained _ hm
    obtain ⟨a, _, ha⟩ := List.mem_map.mp hn
    refine ⟨a, ?_⟩
    unfold maxFrequency
    rw [heq, ha]

theorem indistinguishable_labels_force_error
    {Input Representation : Type} (encode : Input → Representation)
    (predict : Representation → Nat) (x y : Input) (a b : Nat)
    (same : encode x = encode y) (different : a ≠ b) :
    predict (encode x) ≠ a ∨ predict (encode y) ≠ b := by
  by_cases h : predict (encode x) = a
  · right
    intro hb
    apply different
    calc a = predict (encode x) := h.symm
         _ = predict (encode y) := congrArg predict same
         _ = b := hb
  · exact Or.inl h

/- Group identifiers represent equivalence classes, and labels supplies each
   class's recorded labels. Different identifiers may have identical label lists.
   This certificate does not parse inputs or establish a partition. -/
def minimumErrors {Group : Type} (labels : Group → List Nat) : List Group → Nat
  | [] => 0
  | g :: rest => (labels g).length - maxFrequency (labels g) + minimumErrors labels rest

def classifierErrors {Group : Type} (labels : Group → List Nat)
    (groups : List Group) (predict : Group → Nat) : Nat :=
  match groups with
  | [] => 0
  | g :: rest => (labels g).length - (labels g).count (predict g) + classifierErrors labels rest predict

theorem aggregate_error_lower_bound {Group : Type} (labels : Group → List Nat)
    (groups : List Group) (predict : Group → Nat) :
    minimumErrors labels groups ≤ classifierErrors labels groups predict := by
  induction groups with
  | nil => exact Nat.le_refl 0
  | cons g rest ih =>
    exact Nat.add_le_add (class_error_lower_bound (labels g) (predict g)) ih

theorem aggregate_error_bound_attainable {Group : Type} (labels : Group → List Nat)
    (groups : List Group) :
    ∃ predict : Group → Nat, classifierErrors labels groups predict = minimumErrors labels groups := by
  let predict := fun g => Classical.choose (class_error_bound_attainable (labels g))
  have best : ∀ g, (labels g).length - (labels g).count (predict g) =
      (labels g).length - maxFrequency (labels g) :=
    fun g => Classical.choose_spec (class_error_bound_attainable (labels g))
  refine ⟨predict, ?_⟩
  induction groups with
  | nil => rfl
  | cons g rest ih =>
    simp only [classifierErrors, minimumErrors, best g, ih]

#print axioms class_error_lower_bound
#print axioms class_error_bound_attainable
#print axioms indistinguishable_labels_force_error
#print axioms aggregate_error_lower_bound
#print axioms aggregate_error_bound_attainable

end NormalizationAudit
