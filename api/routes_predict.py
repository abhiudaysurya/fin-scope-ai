"""
FinScope AI - Prediction Routes

API endpoints for credit risk predictions:
  - POST /predict     — Single record prediction
  - POST /batch_predict — Batch prediction
"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictionResult,
    ErrorResponse,
    FinancialInput,
    PredictionResult,
)
from services.database import get_db
from services.db_models import PredictionLog, User
from services.prediction_service import PredictionService
from utils.exceptions import ModelInferenceError

router = APIRouter(tags=["Predictions"])

# Singleton prediction service
_prediction_service = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


@router.post(
    "/predict",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict default risk for a single financial record",
)
async def predict(
    input_data: FinancialInput,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict the probability of loan default for a single applicant.

    Returns a risk score (0-1), risk category, and inference metadata.
    """
    try:
        service = get_prediction_service()
        features = input_data.to_feature_dict()
        result = service.predict_single(features)

        # Log prediction to database
        log_entry = PredictionLog(
            user_id=current_user.id,
            model_name=result["model_name"],
            default_probability=result["default_probability"],
            risk_category=result["risk_category"],
            threshold_used=result["threshold"],
            input_features=features,
            inference_time_ms=result.get("inference_time_ms"),
            request_id=result["request_id"],
            ip_address=request.client.host if request.client else None,
        )
        db.add(log_entry)

        return PredictionResult(**result)

    except ModelInferenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict default risk for a batch of financial records",
)
async def batch_predict(
    batch_request: BatchPredictRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Predict default probability for multiple applicants in a single request.

    Accepts up to 1000 records. Optimized for throughput.
    """
    try:
        service = get_prediction_service()
        feature_dicts = [record.to_feature_dict() for record in batch_request.records]
        results = service.predict_batch(feature_dicts, model_name=batch_request.model_name)

        # Log batch predictions
        for result in results:
            log_entry = PredictionLog(
                user_id=current_user.id,
                model_name=result["model_name"],
                default_probability=result["default_probability"],
                risk_category=result["risk_category"],
                threshold_used=result["threshold"],
                input_features=feature_dicts[result["record_index"]],
                request_id=result["request_id"],
                ip_address=request.client.host if request.client else None,
            )
            db.add(log_entry)

        predictions = [BatchPredictionResult(**r) for r in results]

        return BatchPredictResponse(
            predictions=predictions,
            total_records=len(predictions),
            model_name=results[0]["model_name"] if results else "unknown",
        )

    except ModelInferenceError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
