from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, relationship

from app.db.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    frame_id: Mapped[int] = Column(Integer, ForeignKey("frames.id"), nullable=False, index=True)
    object_name: Mapped[str] = Column(String, nullable=False)
    confidence: Mapped[float] = Column(Float, nullable=False)
    bbox_xmin: Mapped[float] = Column(Float, nullable=False)
    bbox_ymin: Mapped[float] = Column(Float, nullable=False)
    bbox_xmax: Mapped[float] = Column(Float, nullable=False)
    bbox_ymax: Mapped[float] = Column(Float, nullable=False)

    frame = relationship("Frame", back_populates="detections")
