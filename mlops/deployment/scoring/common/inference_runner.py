import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(model, df):
    return model.predict(df)


def postprocess(preds) -> dict:
    return {"predictions": preds.tolist()}


def run(raw_data: str, model, preprocess_fn) -> str:
    try:
        payload = json.loads(raw_data)
        df = preprocess_fn(payload)
        preds = predict(model, df)
        result = postprocess(preds)
        return json.dumps(result)
    except Exception as e:
        logger.exception("Inference failed")
        return json.dumps({"error": str(e)})