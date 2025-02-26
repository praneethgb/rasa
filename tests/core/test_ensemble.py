from pathlib import Path
from typing import List, Any, Text, Optional
from _pytest.monkeypatch import MonkeyPatch
import pytest
from _pytest.logging import LogCaptureFixture
import logging
import copy
import numpy as np

from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.events import UserUttered, Event, SlotSet
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.policies.ensemble import (
    PolicyEnsemble,
    InvalidPolicyConfig,
    SimplePolicyEnsemble,
)
from rasa.core.policies.rule_policy import RulePolicy

from rasa.shared.core.events import ActionExecuted, DefinePrevUserUtteredFeaturization
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.core.policies.unexpected_intent_policy import UnexpecTEDIntentPolicy
from rasa.core.agent import Agent
from tests.core import test_utils


def _action_unlikely_intent_for(intent_name: Text):
    _original = UnexpecTEDIntentPolicy.predict_action_probabilities

    def predict_action_probabilities(
        self, tracker, domain, interpreter, **kwargs,
    ) -> PolicyPrediction:
        latest_event = tracker.events[-1]
        if (
            isinstance(latest_event, UserUttered)
            and latest_event.parse_data["intent"]["name"] == intent_name
        ):
            return PolicyPrediction.for_action_name(domain, ACTION_UNLIKELY_INTENT_NAME)
        return _original(self, tracker, domain, interpreter, **kwargs)

    return predict_action_probabilities


class WorkingPolicy(Policy):
    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Policy:
        return WorkingPolicy()

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, WorkingPolicy)


def test_policy_loading_simple(tmp_path: Path):
    original_policy_ensemble = PolicyEnsemble([WorkingPolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    loaded_policy_ensemble = PolicyEnsemble.load(str(tmp_path))
    assert original_policy_ensemble.policies == loaded_policy_ensemble.policies


class ConstantPolicy(Policy):
    def __init__(
        self,
        priority: Optional[int] = None,
        predict_index: Optional[int] = None,
        confidence: float = 1,
        is_end_to_end_prediction: bool = False,
        is_no_user_prediction: bool = False,
        events: Optional[List[Event]] = None,
        optional_events: Optional[List[Event]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(priority=priority, **kwargs)
        self.predict_index = predict_index
        self.confidence = confidence
        self.is_end_to_end_prediction = is_end_to_end_prediction
        self.is_no_user_prediction = is_no_user_prediction
        self.events = events or []
        self.optional_events = optional_events or []

    @classmethod
    def load(cls, args, **kwargs) -> Policy:
        pass

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        result = [0.0] * domain.num_actions
        result[self.predict_index] = self.confidence

        return PolicyPrediction(
            result,
            self.__class__.__name__,
            policy_priority=self.priority,
            is_end_to_end_prediction=self.is_end_to_end_prediction,
            is_no_user_prediction=self.is_no_user_prediction,
            events=self.events,
            optional_events=self.optional_events,
        )


def test_policy_priority():
    domain = Domain.load("data/test_domains/default.yml")
    tracker = DialogueStateTracker.from_events("test", [UserUttered("hi")], [])

    priority_1 = ConstantPolicy(priority=1, predict_index=0)
    priority_2 = ConstantPolicy(priority=2, predict_index=1)

    policy_ensemble_0 = SimplePolicyEnsemble([priority_1, priority_2])
    policy_ensemble_1 = SimplePolicyEnsemble([priority_2, priority_1])

    priority_2_result = priority_2.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )

    i = 1  # index of priority_2 in ensemble_0
    prediction = policy_ensemble_0.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.policy_name == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert prediction.probabilities == priority_2_result.probabilities

    i = 0  # index of priority_2 in ensemble_1
    prediction = policy_ensemble_1.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.policy_name == "policy_{}_{}".format(i, type(priority_2).__name__)
    assert prediction.probabilities == priority_2_result.probabilities


class LoadReturnsNonePolicy(Policy):
    @classmethod
    def load(cls, *args, **kwargs) -> None:
        return None

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass


def test_policy_loading_load_returns_none(tmp_path: Path, caplog: LogCaptureFixture):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsNonePolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    with caplog.at_level(logging.WARNING):
        ensemble = PolicyEnsemble.load(str(tmp_path))
        assert (
            caplog.records.pop().msg
            == "Failed to load policy tests.core.test_ensemble."
            "LoadReturnsNonePolicy: load returned None"
        )
        assert len(ensemble.policies) == 0


class LoadReturnsWrongTypePolicy(Policy):
    @classmethod
    def load(cls, *args, **kwargs) -> Text:
        return ""

    def persist(self, _) -> None:
        pass

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        pass


def test_policy_loading_load_returns_wrong_type(tmp_path: Path):
    original_policy_ensemble = PolicyEnsemble([LoadReturnsWrongTypePolicy()])
    original_policy_ensemble.train([], None, RegexInterpreter())
    original_policy_ensemble.persist(str(tmp_path))

    with pytest.raises(Exception):
        PolicyEnsemble.load(str(tmp_path))


@pytest.mark.parametrize(
    "valid_config",
    [
        {"policy": [{"name": "MemoizationPolicy"}]},
        {"policies": [{"name": "MemoizationPolicy"}]},
    ],
)
def test_valid_policy_configurations(valid_config):
    assert PolicyEnsemble.from_dict(valid_config)


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"police": [{"name": "MemoizationPolicy"}]},
        {"policies": []},
        {"policies": [{"name": "ykaüoppodas"}]},
        {"policy": [{"name": "ykaüoppodas"}]},
        {"policy": [{"name": "ykaüoppodas.bladibla"}]},
    ],
)
def test_invalid_policy_configurations(invalid_config):
    with pytest.raises(InvalidPolicyConfig):
        PolicyEnsemble.from_dict(invalid_config)


def test_from_dict_does_not_change_passed_dict_parameter():
    config = {
        "policies": [
            {
                "name": "TEDPolicy",
                "featurizer": [
                    {
                        "name": "MaxHistoryTrackerFeaturizer",
                        "max_history": 5,
                        "state_featurizer": [{"name": "SingleStateFeaturizer"}],
                    }
                ],
            }
        ]
    }

    config_copy = copy.deepcopy(config)
    PolicyEnsemble.from_dict(config_copy)

    assert config == config_copy


def test_rule_based_data_warnings_no_rule_trackers():
    trackers = [DialogueStateTracker("some-id", slots=[], is_rule_tracker=False)]
    policies = [RulePolicy()]
    ensemble = SimplePolicyEnsemble(policies)

    with pytest.warns(UserWarning) as record:
        ensemble.train(trackers, Domain.empty(), RegexInterpreter())

    assert (
        "Found a rule-based policy in your pipeline but no rule-based training data."
    ) in record[0].message.args[0]


def test_rule_based_data_warnings_no_rule_policy():
    trackers = [DialogueStateTracker("some-id", slots=[], is_rule_tracker=True)]
    policies = [ConstantPolicy()]
    ensemble = SimplePolicyEnsemble(policies)

    with pytest.warns(UserWarning) as record:
        ensemble.train(trackers, Domain.empty(), RegexInterpreter())

    assert (
        "Found rule-based training data but no policy supporting rule-based data."
    ) in record[0].message.args[0]


def test_end_to_end_prediction_supersedes_others(domain: Domain):
    expected_action_index = 2
    expected_confidence = 0.5
    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=100, predict_index=0),
            ConstantPolicy(
                priority=1,
                predict_index=expected_action_index,
                confidence=expected_confidence,
                is_end_to_end_prediction=True,
            ),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    assert prediction.max_confidence == expected_confidence
    assert prediction.max_confidence_index == expected_action_index
    assert prediction.policy_name == f"policy_1_{ConstantPolicy.__name__}"


def test_no_user_prediction_supersedes_others(domain: Domain):
    expected_action_index = 2
    expected_confidence = 0.5
    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=100, predict_index=0),
            ConstantPolicy(priority=1, predict_index=1, is_end_to_end_prediction=True),
            ConstantPolicy(
                priority=1,
                predict_index=expected_action_index,
                confidence=expected_confidence,
                is_no_user_prediction=True,
            ),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    assert prediction.max_confidence == expected_confidence
    assert prediction.max_confidence_index == expected_action_index
    assert prediction.policy_name == f"policy_2_{ConstantPolicy.__name__}"
    assert prediction.is_no_user_prediction
    assert not prediction.is_end_to_end_prediction


def test_prediction_applies_must_have_policy_events(domain: Domain):
    must_have_events = [ActionExecuted("my action")]

    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=10, predict_index=1),
            ConstantPolicy(priority=1, predict_index=2, events=must_have_events),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    # Policy 0 won due to higher prio
    assert prediction.policy_name == f"policy_0_{ConstantPolicy.__name__}"

    # Events of losing policy were applied nevertheless
    assert prediction.events == must_have_events


def test_prediction_applies_optional_policy_events(domain: Domain):
    optional_events = [ActionExecuted("my action")]
    must_have_events = [SlotSet("some slot", "some value")]

    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(
                priority=10,
                predict_index=1,
                events=must_have_events,
                optional_events=optional_events,
            ),
            ConstantPolicy(priority=1, predict_index=2),
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])

    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )

    # Policy 0 won due to higher prio
    assert prediction.policy_name == f"policy_0_{ConstantPolicy.__name__}"

    # Events of losing policy were applied nevertheless
    assert len(prediction.events) == len(optional_events) + len(must_have_events)
    assert all(event in prediction.events for event in optional_events)
    assert all(event in prediction.events for event in must_have_events)


def test_end_to_end_prediction_applies_define_featurization_events(domain: Domain):
    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=100, predict_index=0),
            ConstantPolicy(priority=1, predict_index=1, is_end_to_end_prediction=True),
        ]
    )

    # no events should be added if latest action is not action listen
    tracker = DialogueStateTracker.from_events("test", evts=[])
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.events == []

    # DefinePrevUserUtteredFeaturization should be added after action listen
    tracker = DialogueStateTracker.from_events(
        "test", evts=[ActionExecuted(ACTION_LISTEN_NAME)]
    )
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.events == [DefinePrevUserUtteredFeaturization(True)]


def test_intent_prediction_does_not_apply_define_featurization_events(domain: Domain):
    ensemble = SimplePolicyEnsemble(
        [
            ConstantPolicy(priority=100, predict_index=0),
            ConstantPolicy(priority=1, predict_index=1, is_end_to_end_prediction=False),
        ]
    )

    # no events should be added if latest action is not action listen
    tracker = DialogueStateTracker.from_events("test", evts=[])
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.events == []

    # DefinePrevUserUtteredFeaturization should be added after action listen
    tracker = DialogueStateTracker.from_events(
        "test", evts=[ActionExecuted(ACTION_LISTEN_NAME)]
    )
    prediction = ensemble.probabilities_using_best_policy(
        tracker, domain, RegexInterpreter()
    )
    assert prediction.events == [DefinePrevUserUtteredFeaturization(False)]


@pytest.mark.parametrize(
    "policy_name, confidence, not_in_training_data",
    [
        (f"policy_1_{MemoizationPolicy.__name__}", 1.0, False),
        (f"policy_1_{AugmentedMemoizationPolicy.__name__}", 1.0, False),
        (f"policy_1_{AugmentedMemoizationPolicy.__name__}", None, False),
        (f"policy_1_{RulePolicy.__name__}", 1.0, False),
        ("any", 0.0, True),
    ],
)
def test_is_not_in_training_data(
    policy_name: Text, confidence: Optional[float], not_in_training_data: bool
):
    assert (
        SimplePolicyEnsemble.is_not_in_training_data(policy_name, confidence)
        == not_in_training_data
    )


def test_rule_action_wins_over_action_unlikely_intent(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    unexpected_intent_policy_agent: Agent,
    moodbot_domain: Domain,
):
    # The original training data consists of a rule for `goodbye` intent.
    # We monkey-patch UnexpecTEDIntentPolicy to always predict action_unlikely_intent
    # if last user intent was goodbye. The predicted action from ensemble
    # should be utter_goodbye and not action_unlikely_intent.
    monkeypatch.setattr(
        UnexpecTEDIntentPolicy,
        "predict_action_probabilities",
        _action_unlikely_intent_for("goodbye"),
    )

    tracker = DialogueStateTracker.from_events(
        "rule triggering tracker",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(text="goodbye", intent={"name": "goodbye"}),
        ],
    )
    policy_ensemble = unexpected_intent_policy_agent.policy_ensemble
    prediction = policy_ensemble.probabilities_using_best_policy(
        tracker, moodbot_domain, NaturalLanguageInterpreter()
    )

    test_utils.assert_predicted_action(prediction, moodbot_domain, "utter_goodbye")


def test_ensemble_prevents_multiple_action_unlikely_intents(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    unexpected_intent_policy_agent: Agent,
    moodbot_domain: Domain,
):
    monkeypatch.setattr(
        UnexpecTEDIntentPolicy,
        "predict_action_probabilities",
        _action_unlikely_intent_for("greet"),
    )

    tracker = DialogueStateTracker.from_events(
        "rule triggering tracker",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(text="hello", intent={"name": "greet"}),
            ActionExecuted(ACTION_UNLIKELY_INTENT_NAME),
        ],
    )

    policy_ensemble = unexpected_intent_policy_agent.policy_ensemble
    prediction = policy_ensemble.probabilities_using_best_policy(
        tracker, moodbot_domain, NaturalLanguageInterpreter()
    )

    # prediction cannot be action_unlikely_intent for sure because
    # the last event is not of type UserUttered and that's the
    # first condition for `UnexpecTEDIntentPolicy` to make a prediction
    assert (
        moodbot_domain.action_names_or_texts[np.argmax(prediction.probabilities)]
        != ACTION_UNLIKELY_INTENT_NAME
    )
