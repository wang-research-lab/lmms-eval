from lmms_eval.api.filter import Filter
import re

class LowercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in resps]


class UppercaseFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in resps]


class MapFilter(Filter):
    def __init__(self, mapping_dict: dict = {}, default_value=None) -> None:
        """
        Initializes the MapFilter with a given mapping dictionary and default value.

        Args:
        - mapping_dict (dict): A dictionary containing the key-value mappings.
                               Default is an empty dictionary.
        - default_value (Any): The value to be returned when a key is not found in the mapping_dict.
                               Default is None.

        Example:
        mapper = MapFilter({'A': 1, 'B': 2}, default_value=0)
        """
        assert isinstance(mapping_dict, dict), "Provided mapping_dict is not a dictionary"
        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in resps]

# Updated on July.21, 2024
class TrueFalsetoYesNoFilter(Filter):
    """
    This filter maps True to Yes and False to No. 
    """
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.lower() == "true":
                    resp = "yes"
                elif resp.lower() == "false":
                    resp = "no"
                filtered_resp.append(resp)           
            return filtered_resp
        
        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps
    
class UnanswerableFormatFilter(Filter):
    """
    This filter maps "Unanswerable" to "the answer is: unanswerable".
    """
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp_new = re.sub(r'[^\w\s]', '', resp.lower())
                if resp_new == "unanswerable":
                    resp = "the answer is: unanswerable."
                filtered_resp.append(resp)           
            return filtered_resp
        
        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps

class GQAPretrainLlamaFilter(Filter):
    """
    This filter is specifically designed for parsing answer of pretrained llama on GQA. 
    It clears unexpected "\" and selects direct answer without further details. 
    Example: "computer\"" --> "computer"
    Example: "yes, he is wearing a wetsuit" --> "yes"
    """
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp_split = resp.split(', ')
                if len(resp_split) > 1: 
                    resp = resp_split[0]
                else: 
                    resp = re.sub(r'[^\w\s]', '', resp.lower())
                filtered_resp.append(resp)           
            return filtered_resp
        
        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps  

class VizwizVicunaPretrainFilter(Filter):
    """
    This filter is specifically designed for parsing answer of pretrained vicuna on Vizwiz. 
    Example: "Ъ\n\nUnanswerable." --> "Unanswerable"
    Example: "\nUnanswerable." --> "Unanswerable"
    Example: "a skateboard\".\n\nThe best answer is \"a skateboard\".\n\nThe" --> "a skateboard"
    """
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                resp_split = resp.split('Unanswerable')
                if len(resp_split) > 1: 
                    resp = "Unanswerable\""
                filtered_resp.append(resp)           
            return filtered_resp
        
        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps  